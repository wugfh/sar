import scipy.io as sci
import numpy
import cupy
import matplotlib.pyplot as plt
import time

cupy.cuda.Device(2).use()
start_time = time.time()

## 参数与初始化
data_1 = sci.loadmat("../../../data/English_Bay_ships/data_1.mat")
data_1 = data_1['data_1']
data_1 = cupy.array(data_1, dtype=complex)
c = 299792458                     #光速
Fs = 32317000      #采样率                                   
start = 6.5959e-03         #开窗时间 
Tr = 4.175000000000000e-05        #脉冲宽度                        
f0 = 5.300000000000000e+09                    #载频                     
PRF = 1.256980000000000e+03       #PRF                     
Vr = 7062                       #雷达速度     
B = 30.111e+06        #信号带宽
fc = -6900          #多普勒中心频率
Fa = PRF

# Ka = 1733

wave_lambda = c/f0
Kr = -B/Tr
# Kr = -7.2135e+11
[Na_tmp, Nr_tmp] = cupy.shape(data_1)
[Na, Nr] = cupy.shape(data_1)
data = cupy.zeros([Na+Na, Nr+Nr], dtype=complex)
data[0:Na, 0:Nr] = data_1
[Na,Nr] = cupy.shape(data)


R0 = start*c/2
theta_rc = cupy.arcsin(fc*wave_lambda/(2*Vr))
Ka = 2*Vr**2*cupy.cos(theta_rc)**3/(wave_lambda*R0)
R_eta_c = R0/cupy.cos(theta_rc)
eta_c = 2*Vr*cupy.sin(theta_rc)/wave_lambda

f_tau = cupy.fft.fftshift(cupy.linspace(-Nr/2,Nr/2-1,Nr)*(Fs/Nr))
f_eta = fc + cupy.fft.fftshift(cupy.linspace(-Na/2,Na/2-1,Na)*(Fa/Na))

tau = 2*R_eta_c/c + cupy.linspace(-Nr/2,Nr/2-1, Nr)*(1/Fs)
eta = eta_c + cupy.linspace(-Na/2,Na/2-1,Na)*(1/Fa)

[Ext_time_tau_r, Ext_time_eta_a] = cupy.meshgrid(tau, eta)
[Ext_f_tau, Ext_f_eta] = cupy.meshgrid(f_tau, f_eta)

R_ref = R_eta_c # 将参考目标设为场景中心
# 定义并行计算的核函数
kernel_code = '''
extern "C" 
#define M_PI 3.14159265358979323846
__global__ void sinc_interpolation(
    const double* echo_ftau_feta,
    const int* delta_int,
    const double* delta_remain,
    double* echo_ftau_feta_stolt,
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
                predict_value += echo_ftau_feta[i * Nr + (Nr - 1)] * sinc_y;
            } else if (index < 0) {
                predict_value += echo_ftau_feta[i * Nr] * sinc_y;
            } else {
                predict_value += echo_ftau_feta[i * Nr + index] * sinc_y;
            }
        }
        echo_ftau_feta_stolt[i * Nr + j] = predict_value/sum_sinc;
    }
}
'''

def stolt_interpolation(echo_ftau_feta, delta_int, delta_remain, Na, Nr, sinc_N):
    module = cupy.RawModule(code=kernel_code)
    sinc_interpolation = module.get_function('sinc_interpolation')
    echo_ftau_feta = cupy.ascontiguousarray(echo_ftau_feta)
    # 初始化数据
    echo_ftau_feta_stolt_real = cupy.zeros((Na, Nr), dtype=cupy.double)
    echo_ftau_feta_stolt_imag = cupy.zeros((Na, Nr), dtype=cupy.double)
    echo_ftau_feta_real = cupy.real(echo_ftau_feta).astype(cupy.double)
    echo_ftau_feta_imag = cupy.imag(echo_ftau_feta).astype(cupy.double)

    # 设置线程和块的维度
    threads_per_block = (16, 16)
    blocks_per_grid = (int(cupy.ceil(Na / threads_per_block[0])), int(cupy.ceil(Nr / threads_per_block[1])))

    # 调用核函数
    sinc_interpolation(
        (blocks_per_grid[0], blocks_per_grid[1]), (threads_per_block[0], threads_per_block[1]),
        (echo_ftau_feta_real, delta_int, delta_remain, echo_ftau_feta_stolt_real, Na, Nr, sinc_N)
    )

    sinc_interpolation(
        (blocks_per_grid[0], blocks_per_grid[1]), (threads_per_block[0], threads_per_block[1]),
        (echo_ftau_feta_imag, delta_int, delta_remain, echo_ftau_feta_stolt_imag, Na, Nr, sinc_N)
    )
    echo_ftau_feta_stolt_strip = echo_ftau_feta_stolt_real + 1j * echo_ftau_feta_stolt_imag
    return echo_ftau_feta_stolt_strip

def  strip_focusing(echo_strip):
    ## RFM
    echo_ftau_feta = (cupy.fft.fft2(echo_strip))
    Na, Nr = cupy.shape(echo_strip)
    f_tau = cupy.fft.fftshift((cupy.arange(-Nr/2, Nr/2) * Fs / Nr))
    f_eta = fc + ((cupy.arange(-Na/2, Na/2) * PRF / Na))
    mat_ftau, mat_feta = cupy.meshgrid(f_tau, f_eta)
    
    H3_strip = cupy.exp((4j*cupy.pi*R_ref/c)*cupy.sqrt((f0+mat_ftau)**2 - c**2 * mat_feta**2 / (4*Vr**2)) + 1j*cupy.pi*mat_ftau**2/Kr)
    echo_ftau_feta = echo_ftau_feta * H3_strip

    ## modified stolt mapping
    # map_f_tau = cupy.sqrt((mat_f_tau_stolt+cupy.sqrt(f**2 - c**2 * mat_f_eta_stolt**2 / (4 * vr**2)))**2 + c**2 * mat_f_eta_stolt**2 / (4 * vr**2)) - f
    map_f_tau = cupy.sqrt((f0+mat_ftau)**2+c**2*mat_feta**2/(4*Vr**2))-f0
    delta = (map_f_tau - mat_ftau)/(Fs/Nr) #频率转index
    delta_int = cupy.floor(delta).astype(cupy.int32)
    delta_remain = delta-delta_int

    ## sinc interpolation kernel length, used by stolt mapping
    sinc_N = 8
    echo_ftau_feta_stolt_strip = stolt_interpolation(echo_ftau_feta, delta_int, delta_remain, Na, Nr, sinc_N)
    echo_ftau_feta_stolt_strip = echo_ftau_feta_stolt_strip * cupy.exp(-4j*cupy.pi*mat_ftau*R_ref/c)

    echo_stolt = cupy.fft.ifft2(echo_ftau_feta_stolt_strip)
    echo_no_stolt = cupy.fft.ifft2(echo_ftau_feta)
    return echo_stolt, echo_no_stolt

data_final, _ = strip_focusing(data)
#简单的后期处理
data_final = cupy.abs(data_final)/cupy.max(cupy.max(cupy.abs(data_final)))
data_final = 20*cupy.log10(data_final+1)
data_final = data_final**0.4
data_final = cupy.abs(data_final)/cupy.max(cupy.max(cupy.abs(data_final)))

plt.imshow(abs(cupy.asnumpy(data_final)), cmap='gray')
plt.savefig("../../../fig/nicolas/wk_sim.png", dpi=300)

end_time = time.time()
print('Time:', end_time-start_time)

