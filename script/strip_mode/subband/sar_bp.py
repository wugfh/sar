import numpy as np
import cupy as cp             
import matplotlib.pyplot as plt

class BpFocus:
        # 定义并行计算的核函数
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
    def __init__(self):
        self.c = 299792458                      #光速
        self.Fs = 32317000                      #采样率              
        self.start = 6.5959e-03                 #开窗时间 
        self.Tr = 4.175000000000000e-05         #脉冲宽度                        
        self.f0 = 5.300000000000000e+09         #载频                     
        self.PRF = 1.256980000000000e+03        #PRF                     
        self.Vr = 7062                          #雷达速度     
        self.B = 30.111e+06                     #信号带宽
        self.fc = -6900                         #多普勒中心频率
        self.R0 = self.start*self.c/2
        self.lambda_= self.c/self.f0
        self.theta_c = cp.arcsin(self.fc*self.lambda_/(2*self.Vr))
        self.La = 15
        self.Ta = 2
        self.Kr = self.B/self.Tr
        self.Nr = int(cp.ceil(self.Fs*self.Tr))
        self.Na = int(cp.ceil(self.PRF*self.Ta))
        self.points_n = 5
        self.points_r = self.R0+cp.linspace(-500, 500, self.points_n)
        self.points_a = cp.linspace(-1000, 1000, self.points_n)

    def echo_generate(self):
        Rc = self.R0/cp.cos(self.theta_c)
        tau = 2*Rc/self.c + cp.arange(-self.Nr/2, self.Nr/2, 1)*(1/self.Fs)
        eta_c = -Rc*cp.sin(self.theta_c)/self.Vr
        eta = eta_c + cp.arange(-self.Na/2, self.Na/2, 1)*(1/self.PRF)  
        mat_tau, mat_eta = cp.meshgrid(tau, eta)
        S_echo = cp.zeros((self.Na, self.Nr), dtype=cp.complex128)
        for i in range(self.points_n):
            R0_tar = self.points_r[i]/cp.cos(self.theta_c)
            R_eta = cp.sqrt(R0_tar**2 + (self.Vr*mat_eta - self.points_a[i])**2)
            Wr = cp.abs(mat_tau-2*R_eta/self.c)<self.Tr/2
            Tstrip_tar = 0.886*self.lambda_*R0_tar/(self.La*self.Vr*cp.cos(self.theta_c)**2)
            Wa =  cp.abs(mat_eta-(self.points_a[i]/self.Vr + eta_c)) < Tstrip_tar/2
            # Wa = cp.sinc(self.La*(cp.arccos(R0_tar/R_eta)-self.theta_c)/self.lambda_)**2
            Phase = cp.exp(-4j*cp.pi*R_eta/self.lambda_)*cp.exp(1j*cp.pi*self.Kr*(mat_tau-2*R_eta/self.c)**2)
            S_echo += Wr*Wa*Phase
        return S_echo

    def Bp_preprocess(self, S_echo):
        f_tau = cp.fft.fftshift(cp.arange(-self.Nr/2, self.Nr/2, 1)*(self.Fs/self.Nr))
        f_eta = cp.fft.fftshift(self.fc + cp.arange(-self.Na/2, self.Na/2, 1)*(self.PRF/self.Na))
        mat_f_tau, mat_f_eta = cp.meshgrid(f_tau, f_eta)
        Hr = cp.exp(1j*cp.pi*mat_f_tau**2/self.Kr)
        S_ftau_eta = cp.fft.fft(S_echo, axis=1)
        S_ftau_eta = S_ftau_eta*Hr
        echo = cp.fft.ifft(S_ftau_eta, axis=1)
        return echo
    
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

    def Bp_foucs(self, echo):
        Rc = self.R0/cp.cos(self.theta_c)
        tau = 2*Rc/self.c + cp.arange(-self.Nr/2, self.Nr/2, 1)*(1/self.Fs)
        eta_c = -Rc*cp.sin(self.theta_c)/self.Vr
        eta = eta_c + cp.arange(-self.Na/2, self.Na/2, 1)*(1/self.PRF)  
        mat_tau, mat_eta = cp.meshgrid(tau, eta)
        mat_R = mat_tau*self.c/2
        output = cp.zeros((self.Na, self.Nr), dtype=cp.complex128)
        print(cp.min(cp.min(mat_R)), cp.max(cp.max(mat_R)))
        print(Rc)

        for i in range(self.Na):
            eta_now = (i-self.Na/2)/self.PRF+eta_c
            R_eta = cp.sqrt(mat_R**2 + (self.Vr*(mat_eta-eta_now))**2)
            delta_t = 2*(R_eta-mat_R)/self.c
            delta = delta_t/(1/self.Fs)
            output += self.sinc_interpolation(echo, delta, self.Na, self.Nr, 8)*cp.exp(4j*cp.pi*R_eta/self.lambda_) 
        return output
    
if __name__ == '__main__':
    bp = BpFocus()
    echo = bp.echo_generate()
    plt.figure(1)
    plt.imshow(cp.abs(echo).get(), aspect="auto")
    plt.savefig("../../../fig/bp/echo.png", dpi=300)

    echo_pre = bp.Bp_preprocess(echo)
    plt.figure(2)
    plt.imshow(cp.abs(echo_pre).get(), aspect="auto")
    plt.savefig("../../../fig/bp/preprocess.png", dpi=300)

    output = bp.Bp_foucs(echo_pre)
    plt.figure(3)
    plt.imshow(cp.abs(output).get(), aspect="auto")
    plt.savefig("../../../fig/bp/bp_dot_result.png", dpi=300)

            


