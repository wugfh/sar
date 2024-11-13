import scipy.io as sci
import numpy
import cupy
import matplotlib.pyplot as plt
import time

start_time = time.time()

## 参数与初始化
data_1 = sci.loadmat("../../data/English_Bay_ships/data_1.mat")
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

## 一致RCMC
data = data*cupy.exp(-2j*cupy.pi*fc*Ext_time_eta_a)
data_tau_feta = cupy.fft.fft(data, Na, axis = 0) # 首先变换到距离多普勒域

D = cupy.sqrt(1-c**2*Ext_f_eta**2/(4*Vr**2*f0**2))#徙动因子
D_ref = cupy.sqrt(1-c**2*fc**2/(4*Vr**2*f0**2)) # 参考频率处的徙动因子（方位向频率中心）
#大斜视角下，距离调频率随距离变化
K_factor = c*R0*Ext_f_eta**2/(2*Vr**2*f0**3*D**3)
Km = Kr/(1-Kr*K_factor) 

data_ftau_feta = cupy.fft.fft(data_tau_feta, Nr, axis=1)
H_rfm = cupy.exp(-4j*cupy.pi*(R0-R_ref)/c*cupy.sqrt((f0+Ext_f_tau)**2-c**2*Ext_f_eta**2/(4*Vr**2)))
data_ftau_feta = data_ftau_feta*H_rfm #一致rcmc

## stolt 插值,残余RCMC
# Ext_f_tau_new = sqrt((f0+Ext_f_tau).**2-c**2*Ext_f_eta.**2/(4*Vr**2))-f0 #对应的stolt频谱，但是频率非线性变化
# max_f_stolt = max(max(Ext_f_tau_new))
# min_f_stolt = min(min(Ext_f_tau_new))
# f_stolt = fftshift((0:Nr-1)*(max_f_stolt-min_f_stolt))/Nr+min_f_stolt
f_stolt = cupy.fft.fftshift(cupy.linspace(-Nr/2,Nr/2-1,Nr))*(Fs/Nr) # 构造线性变化的stolt频率轴

[Ext_f_stolt, Ext_f_eta] = cupy.meshgrid(f_stolt, f_eta)
Ext_map_f_tau = cupy.sqrt((f0+Ext_f_stolt)**2+c**2*Ext_f_eta**2/(4*Vr**2))-f0 #线性变化的stolt频率轴与原始频率轴的对应（stolt 映射）

Ext_map_f_pos = (Ext_map_f_tau)/(Fs/Nr) #频率转index
Ext_map_f_pos = (Ext_map_f_pos<0)*Nr + Ext_map_f_pos
Ext_map_f_int = cupy.floor(Ext_map_f_pos)
Ext_map_f_remain = Ext_map_f_pos-Ext_map_f_int

#插值使用8位 sinc插值
sinc_N = 8

# 定义并行计算的核函数
kernel_code = '''
extern "C" 
#define M_PI 3.14159265358979323846
__global__ void sinc_interpolation(
    const double* data_ftau_feta,
    const double* Ext_map_f_int,
    const double* Ext_map_f_remain,
    double* data_ftau_feta_stolt,
    double* check_matrix,
    int Na, int Nr, int sinc_N) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < Na && j < Nr) {
        double map_f_int = Ext_map_f_int[i * Nr + j];
        double map_f_remain = Ext_map_f_remain[i * Nr + j];
        double predict_value = 0;
        check_matrix[i * Nr + j] = sinc_N/2;
        for (int m = 0; m < sinc_N; ++m) {
            double sinc_x = map_f_remain - (m - sinc_N/2);
            double sinc_y = sin(M_PI * sinc_x) / (M_PI * sinc_x);
            int index = map_f_int + m - sinc_N/2;
            if (index >= Nr) {
                predict_value += data_ftau_feta[i * Nr + (Nr - 1)] * sinc_y;
            } else if (index < 0) {
                predict_value += data_ftau_feta[i * Nr] * sinc_y;
            } else {
                predict_value += data_ftau_feta[i * Nr + index] * sinc_y;
            }
        }
        data_ftau_feta_stolt[i * Nr + j] = predict_value;
    }
}
'''
# 编译核函数
module = cupy.RawModule(code=kernel_code)
sinc_interpolation = module.get_function('sinc_interpolation')

# 初始化数据
data_ftau_feta_stolt_real = cupy.zeros((Na, Nr), dtype=cupy.double)
data_ftau_feta_stolt_imag = cupy.zeros((Na, Nr), dtype=cupy.double)
data_ftau_feta_real = cupy.real(data_ftau_feta).astype(cupy.double)
data_ftau_feta_imag = cupy.imag(data_ftau_feta).astype(cupy.double)
check_matrix = cupy.zeros((Na, Nr), dtype=cupy.double)

# 设置线程和块的维度
threads_per_block = (16, 16)
blocks_per_grid = (int(cupy.ceil(Na / threads_per_block[0])), int(cupy.ceil(Nr / threads_per_block[1])))

# 调用核函数
sinc_interpolation(
    (blocks_per_grid[0], blocks_per_grid[1]), (threads_per_block[0], threads_per_block[1]),
    (data_ftau_feta_real, Ext_map_f_int, Ext_map_f_remain, data_ftau_feta_stolt_real, check_matrix, Na, Nr, sinc_N)
)

sinc_interpolation(
    (blocks_per_grid[0], blocks_per_grid[1]), (threads_per_block[0], threads_per_block[1]),
    (data_ftau_feta_imag, Ext_map_f_int, Ext_map_f_remain, data_ftau_feta_stolt_imag, check_matrix, Na, Nr, sinc_N)
)

data_ftau_feta_stolt = data_ftau_feta_stolt_real + 1j * data_ftau_feta_stolt_imag


## 成像
Hr = cupy.exp(1j*cupy.pi*(D/(Km*D_ref))*Ext_f_stolt**2) 
data_ftau_feta_stolt = data_ftau_feta_stolt*Hr # 距离压缩

data_tau_feta = cupy.fft.ifft(data_ftau_feta_stolt, Nr, axis=1)
R0_RCMC = c*Ext_time_tau_r/2
Ha = cupy.exp(4j*cupy.pi*D*R0_RCMC*f0/c) 
data_tau_feta = data_tau_feta*Ha # 方位压缩
data_final = cupy.fft.ifft(data_tau_feta, Na, axis=0)

#简单的后期处理
data_final = cupy.abs(data_final)/cupy.max(cupy.max(cupy.abs(data_final)))
data_final = 20*cupy.log10(data_final+1)
data_final = data_final**0.4
data_final = cupy.abs(data_final)/cupy.max(cupy.max(cupy.abs(data_final)))

plt.imshow(abs(cupy.asnumpy(data_final)), cmap='gray')
# plt.savefig("../../fig/nicolas/wk_sim.png", dpi=300)

end_time = time.time()
print('Time:', end_time-start_time)
plt.show()
