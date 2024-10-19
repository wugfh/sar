import scipy.io as sci
import numpy
import cupy
import matplotlib.pyplot as plt

data_1 = sci.loadmat("../../data/English_Bay_ships/data_1.mat")
data_1 = data_1['data_1']

data_1 = cupy.array(data_1, dtype=complex)
c = 299792458                     #光速
Fs = 32317000      #采样率

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
f_eta = fc + (cupy.linspace(-Na/2,Na/2-1,Na)*(Fa/Na))

tau = 2*R_eta_c/c + cupy.linspace(-Nr/2,Nr/2-1, Nr)*(1/Fs)
eta = eta_c + cupy.linspace(-Na/2,Na/2-1,Na)*(1/Fa)

[Ext_time_tau_r, Ext_time_eta_a] = cupy.meshgrid(tau, eta)
[Ext_f_tau, Ext_f_eta] = cupy.meshgrid(f_tau, f_eta)


## 范围压缩
mat_D = cupy.sqrt(1-c**2*Ext_f_eta**2/(4*Vr**2*f0**2))#徙动因子
Ksrc = 2*Vr**2*f0**3*mat_D**3/(c*R0*Ext_f_eta**2)

data_fft_r = cupy.fft.fft(data, Nr, axis = 1) 
Hr = cupy.exp(1j*cupy.pi*Ext_f_tau**2/Kr)
Hm = cupy.exp(-1j*cupy.pi*Ext_f_tau**2/Ksrc)
data_fft_cr = data_fft_r*Hr*Hm
data_cr = cupy.fft.ifft(data_fft_cr, Nr, axis = 1)

## RCMC
data_fft_a = cupy.fft.fft(data_cr, Na, axis=0)
sinc_N = 8
mat_R0 = Ext_time_tau_r*c/2;  

sinc_kernel_code = '''
extern "C" 
#define M_PI 3.14159265358979323846
#define c_speed 299792458
#define Fs 32317000
#define sinc_N 8
__global__ void sinc_interpolation(
    const double *data_fft_a, 
    const double* mat_D, 
    const double *mat_R0, 
    double *data_fft_rcmc, 
    double *check_matrix, 
    int Na, int Nr)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < Na && j < Nr)
    {   
        check_matrix[i*Nr+j] = data_fft_a[i*Nr+j];
        double dR = mat_R0[i*Nr+j]/mat_D[i*Nr+j] - mat_R0[i*Nr+j];
        dR = dR*2/(c_speed/Fs);
        double dR_int = floor(dR);
        double dR_remain = dR - dR_int;
        double predict_value = 0;
        for(int m = 0; m < sinc_N; m++)
        {
            int index = j+dR_int+m-sinc_N/2;
            double sinc_x = dR_remain - (m-sinc_N/2);
            double sinc_value = sin(M_PI * sinc_x) / (M_PI * sinc_x);
            if(index>=Nr){
                predict_value += data_fft_a[i*Nr+Nr-1]*sinc_value;
            }
            else if(index<0){
                predict_value += data_fft_a[i*Nr]*sinc_value;
            }
            else{
                predict_value += data_fft_a[i*Nr+index]*sinc_value;
            }
        }
        data_fft_rcmc[i*Nr+j] = predict_value;
    }
}
'''

module = cupy.RawModule(code=sinc_kernel_code)
sinc_interpolation = module.get_function("sinc_interpolation")

data_fft_a = cupy.ascontiguousarray(data_fft_a)
data_fft_a_rcmc_real = cupy.zeros((Na, Nr), dtype=cupy.double)
data_fft_a_rcmc_imag = cupy.zeros((Na, Nr), dtype=cupy.double)
data_fft_a_real = cupy.real(data_fft_a).astype(cupy.double)
data_fft_a_imag = cupy.imag(data_fft_a).astype(cupy.double)
check_matrix = cupy.zeros((Na, Nr), dtype=cupy.double)

thread_per_block = (16, 16)
block_per_grid = (int(cupy.ceil(Na/thread_per_block[0])), int(cupy.ceil(Nr/thread_per_block[1])))

sinc_interpolation((block_per_grid[0], block_per_grid[1]), (thread_per_block[0], thread_per_block[1]), 
            (data_fft_a_real, mat_D, mat_R0, data_fft_a_rcmc_real, check_matrix, Na, Nr))

sinc_interpolation((block_per_grid[0], block_per_grid[1]), (thread_per_block[0], thread_per_block[1]), 
            (data_fft_a_imag, mat_D, mat_R0, data_fft_a_rcmc_imag, check_matrix, Na, Nr))


print(check_matrix-data_fft_a_imag)
data_fft_a_rcmc = data_fft_a_rcmc_real + 1j*data_fft_a_rcmc_imag

## 方位压缩
Ha = cupy.exp(4j*cupy.pi*mat_D*mat_R0*f0/c)
offset = cupy.exp(2j*cupy.pi*Ext_f_eta*eta_c)
data_fft_a_rcmc = data_fft_a_rcmc*Ha*offset
data_ca_rcmc = cupy.fft.ifft(data_fft_a_rcmc, Na, axis=0)

data_final = data_ca_rcmc


data_final = cupy.abs(data_final)/cupy.max(cupy.max(cupy.abs(data_final)))
data_final = 20*cupy.log10(data_final+1)
data_final = data_final**0.4
data_final = cupy.abs(data_final)/cupy.max(cupy.max(cupy.abs(data_final)))


plt.imshow(abs(cupy.asnumpy(data_final)), cmap='gray')
plt.savefig("../../fig/nicolas/m_chan_test1.png", dpi=300)