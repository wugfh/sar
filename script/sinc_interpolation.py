import numpy as np
import cupy as cp
from scipy.interpolate import interp1d
import tqdm

_Nk = None
_tick = None
_sinc_interpolator = None

def sinc_interp_v2(input_array, coord):
    """
    8点汉明窗sinc插值
    :param input_array: 一维输入信号 (numpy数组)
    :param coord: 插值坐标数组 
    :return: 插值结果
    """
    global _Nk, _tick, _sinc_interpolator

    # ===================== 持久化缓存：仅首次运行初始化 =====================
    if _sinc_interpolator is None:
        _Nk = 8          # 8点插值核
        _tick = 16       # 核采样步长
        # 1. 生成sinc核横坐标
        sinc_x = np.arange(-_Nk/2, _Nk/2 + 1/_tick, 1/_tick)  # 对齐MATLAB: -4:1/16:4
        # 2. 计算sinc函数值
        sinc_table = np.sinc(sinc_x)
        # 3. 汉明窗加权
        hamming_win = np.hamming(len(sinc_table))
        sinc_table = hamming_win * sinc_table
        # 4. 按tick分组归一化
        for i in range(_tick):
            segment = sinc_table[i::_tick]
            sinc_table[i::_tick] = segment / np.sum(segment)
        # 5. 首尾元素对齐
        sinc_table[-1] = sinc_table[0]
        # 6. 创建最近邻插值器
        _sinc_interpolator = interp1d(
            sinc_x, sinc_table,
            kind='nearest',    # 最近邻插值
            fill_value="extrapolate",  # 外推保持一致
            bounds_error=False
        )

    # ===================== 核心插值逻辑 =====================
    N_col = len(input_array)
    # 初始化输出，类型与输入一致
    output = np.zeros_like(coord, dtype=input_array.dtype)
    
    # 索引计算
    idx_input = np.floor(coord).astype(np.int32) - _Nk // 2  # floor(coord)-4
    idx_weight = idx_input - coord

    # 输入信号最近邻插值
    def input_nearest(x):
        # 1-based索引 → 0-based，裁剪边界防止越界
        x_idx = np.round(x).astype(np.int32) - 1
        x_idx = np.clip(x_idx, 0, N_col - 1)
        return input_array[x_idx]

    # 8点核循环累加
    for _ in tqdm.tqdm(range(_Nk), desc="Sinc Interpolation"):
        weight = _sinc_interpolator(idx_weight)
        val = input_nearest(idx_input)
        output += weight * val
        # 索引步进
        idx_input += 1
        idx_weight += 1

    # 越界坐标置0
    out_of_bounds = (coord < 1) | (coord > N_col)
    output[out_of_bounds] = 0

    return output

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
        if not isinstance(in_data, cp.ndarray):
            return self.sinc_interpolation_cpu(in_data, delta, Na, Nr, sinc_N)
        
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
    
    def sinc_interpolation_cpu(self, in_data, delta, Na, Nr, sinc_N):

        out_data = sinc_interp_v2(in_data, delta+np.tile(np.arange(Nr)[None, :], (Na, 1)))

        return out_data
    