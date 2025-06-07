import numpy as np
import cupy as cp

class SincInterpolation:
    kernel_code = '''
    extern "C" 
    #define M_PI 3.14159265358979323846
    __global__ void sinc_interpolation(
        const double* in_data,
        const int* delta_int,
        const double* delta_remain,
        double* out_data,
        int Na, int Nr, int sinc_N) {

        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        
        if (i < Na && j < Nr) {
            int  del_int = delta_int[i * Nr + j];
            double del_remain = delta_remain[i * Nr + j];
            double predict_value = 0;
            double sum_sinc = 0;
            for (int m = 0; m < sinc_N; ++m) {
                double sinc_x = del_remain - (m - sinc_N/2);
                double sinc_y = sin(M_PI * sinc_x) / (M_PI * sinc_x);
                if(sinc_x < 1e-6 && sinc_x > -1e-6) {
                    sinc_y = 1;
                }
                int index = del_int + j + m - sinc_N/2;
                sum_sinc += sinc_y;
                if (index >= Nr) {
                    predict_value += 0;
                } else if (index < 0) {
                    predict_value += 0;
                } else {
                    predict_value += in_data[i * Nr + index] * sinc_y;
                }
            }
            out_data[i * Nr + j] = predict_value/sum_sinc;
        }
    }
    '''

    def sinc_interpolation(self, in_data, delta, Na, Nr, sinc_N):
        delta_int = cp.floor(delta).astype(cp.int32)
        delta_remain = delta-delta_int
        module = cp.RawModule(code=self.kernel_code)
        sinc_interpolation = module.get_function('sinc_interpolation')
        in_data = cp.ascontiguousarray(in_data)
        # 初始化数据
        out_data_real = cp.zeros((Na, Nr), dtype=cp.double)
        out_data_imag = cp.zeros((Na, Nr), dtype=cp.double)
        in_data_real = cp.real(in_data).astype(cp.double)
        in_data_imag = cp.imag(in_data).astype(cp.double)

        # 设置线程和块的维度
        threads_per_block = (16, 16)
        blocks_per_grid = (int(cp.ceil(Na / threads_per_block[0])), int(cp.ceil(Nr / threads_per_block[1])))

        # 调用核函数
        sinc_interpolation(
            (blocks_per_grid[0], blocks_per_grid[1]), (threads_per_block[0], threads_per_block[1]),
            (in_data_real, delta_int, delta_remain, out_data_real, Na, Nr, sinc_N)
        )

        sinc_interpolation(
            (blocks_per_grid[0], blocks_per_grid[1]), (threads_per_block[0], threads_per_block[1]),
            (in_data_imag, delta_int, delta_remain, out_data_imag, Na, Nr, sinc_N)
        )
        out_data = out_data_real + 1j * out_data_imag
        return out_data