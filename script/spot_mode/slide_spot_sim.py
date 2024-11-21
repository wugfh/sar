import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

cp.cuda.Device(2).use()
# 定义参数
f = 5.6e9  # 载波频率
La = 6
PRF = 2318
Tr = 4e-6
Br = 150e6
Fr = 200e6
vr = 7200
Rc = 600e3
omega = cp.deg2rad(0.2656)  # 波束旋转速度
theta_c = cp.deg2rad(15)
c = 299792458

R0 = Rc * cp.cos(theta_c)
lambda_ = c / f
Kr = Br / Tr
Nr = int(cp.ceil(1.2*Fr * Tr))

A = 1 - omega * R0 / (vr * cp.cos(theta_c)**2) # 放缩因子
Ta =  0.886*Rc*lambda_/(A*La*vr*cp.cos(theta_c))
# Ta = 12
Na = int(cp.ceil(PRF * Ta))
eta_c_strip = -R0 * cp.tan(theta_c) / vr
Bf = 2*0.886*vr*cp.cos(theta_c)/La
Bsq = 2*vr*Br*cp.sin(theta_c)/c     

def equation(x):
    return -R0.get() * np.tan(theta_c.get() - omega.get()*x) / vr - x

eta_c_spot = fsolve(equation, eta_c_strip.get()) # 解出滑动聚焦的景中心时间
print("strip center time, slide spot center time", eta_c_strip, eta_c_spot)
eta_c_spot = cp.array(eta_c_spot)

tau_strip = 2 * cp.sqrt(R0**2+vr**2 * eta_c_strip**2) / c + cp.arange(-Nr/2, Nr/2) / Fr
tau_spot = 2 *cp.sqrt(R0**2+vr**2 * eta_c_spot**2) / c + cp.arange(-Nr/2, Nr/2) / Fr 
eta_strip = eta_c_strip + cp.arange(-Na/2, Na/2) / PRF
eta_spot = eta_c_spot + cp.arange(-Na/2, Na/2) / PRF

feta_c = 2 * vr * cp.sin(theta_c) / lambda_

mat_tau_strip, mat_eta_strip = cp.meshgrid(tau_strip, eta_strip)
mat_tau_spot, mat_eta_spot = cp.meshgrid(tau_spot, eta_spot)

point_n = 1
point_x = cp.linspace(-2000, 2000, point_n)
point_y = cp.linspace(-2000, 2000, point_n)

S_echo_spot = cp.zeros((Na, Nr), dtype=cp.complex64)
S_echo_strip = cp.zeros((Na, Nr), dtype=cp.complex64)


## generate echo
for i in range(point_n):
    # slide spot
    R0_tar = cp.sqrt(point_x[i]**2 + R0**2)
    R_eta_spot = cp.sqrt(R0_tar**2 + (vr * mat_eta_spot - point_y[i])**2)
    Wr_spot = (cp.abs(mat_tau_spot - 2 * R_eta_spot / c) <= Tr / 2)
    Wa_spot = cp.sinc(La * (cp.arccos(R0_tar / R_eta_spot) - (theta_c - omega * mat_eta_spot)) / lambda_)**2
    Phase_spot = cp.exp(-4j * cp.pi * f * R_eta_spot / c) * cp.exp(1j * cp.pi * Kr * (mat_tau_spot - 2 * R_eta_spot / c)**2)
    S_echo_spot += Wr_spot * Wa_spot * Phase_spot

    # strip
    R_eta_strip = cp.sqrt(R0_tar**2 + (vr * mat_eta_strip - point_y[i])**2)
    Wr_strip = (cp.abs(mat_tau_strip - 2 * R_eta_strip / c) <= Tr / 2)
    Wa_strip = cp.sinc(La * (cp.arccos(R0_tar / R_eta_strip) - theta_c) / lambda_)**2
    Phase_strip = cp.exp(-4j * cp.pi * f * R_eta_strip / c) * cp.exp(1j * cp.pi * Kr * (mat_tau_strip - 2 * R_eta_strip / c)**2)
    S_echo_strip += Wr_strip * Wa_strip * Phase_strip

S_tau_feta_spot = cp.fft.fft(S_echo_spot, Na, axis=0)
S_tau_feta_strip = cp.fft.fft(S_echo_strip, Na, axis=0)

S_ftau_feta_strip = cp.fft.fft2(S_echo_strip)
S_ftau_feta_spot = cp.fft.fft2(S_echo_spot)

plt.figure(1)
plt.subplot(1, 2, 1)
plt.imshow(cp.abs(S_tau_feta_spot).get(), aspect='auto')
plt.title("slide spotlight")
plt.subplot(1, 2, 2)
plt.imshow(cp.abs(S_tau_feta_strip).get(), aspect='auto')
plt.title("strip")
plt.savefig("../../fig/slide_spot/slide_spot_strip_feta.png", dpi=300)

plt.figure(2)
plt.subplot(1, 2, 1)
plt.imshow(cp.abs(S_echo_spot).get(), aspect='auto')
plt.title("slide spotlight")
plt.subplot(1, 2, 2)
plt.imshow(cp.abs(S_echo_strip).get(), aspect='auto')
plt.title("strip")
plt.savefig("../../fig/slide_spot/slide_spot_strip.png", dpi=300)

plt.figure(3)
plt.subplot(1, 2, 1)
plt.imshow(cp.abs(cp.fft.fftshift(S_ftau_feta_spot)).get(), aspect='auto')
plt.title("slide spotlight")
plt.subplot(1, 2, 2)
plt.imshow(cp.abs(cp.fft.fftshift(S_ftau_feta_strip)).get(), aspect='auto')
plt.title("strip")
plt.savefig("../../fig/slide_spot/slide_spot_strip_fft2.png", dpi=300)

# azimuth dramping, convolve with the signal simliar to transmit signal
# remove the doppler centoid varying 
R_rot = R0/(1-A)
k_rot = -2 * vr**2 * cp.cos(theta_c)**3 / (lambda_ * R_rot)

H1 = cp.exp(-1j * cp.pi * k_rot * mat_eta_spot**2 - 1j * cp.pi * feta_c * mat_eta_spot)
echo = S_echo_spot * H1

echo_ftau_feta = (cp.fft.fft2(echo))
plt.figure(4)
plt.subplot(1, 2, 1)
plt.imshow(cp.abs(cp.fft.fftshift(S_ftau_feta_spot, axes=1)).get(), aspect='auto')
plt.title("before deramping")
plt.subplot(1, 2, 2)
plt.imshow(cp.abs(cp.fft.fftshift(echo_ftau_feta, axes=1)).get(), aspect='auto')
plt.title("after deramping")
plt.savefig("../../fig/slide_spot/slide_spot_preprocess_fft2.png", dpi=300)

## azimuth mosaic
copy_cnt = int(2*cp.ceil((Bf+Bsq)/(2*PRF)) + 1)
Na_up = Na*copy_cnt
print("copy count:", copy_cnt)
echo_mosaic = cp.zeros((Na_up, Nr), dtype=cp.complex64)
for i in range(copy_cnt):
    echo_mosaic[i*Na:(i+1)*Na, :] = echo_ftau_feta
plt.figure(4)
plt.subplot(1, 2, 1)
plt.imshow(cp.abs(cp.fft.fftshift(echo_ftau_feta, axes=1)).get(), aspect='auto')
plt.title("on copy")
plt.subplot(1, 2, 2)
plt.imshow(cp.abs(cp.fft.fftshift(echo_mosaic, axes=1)).get(), aspect='auto')
plt.title("mosaic result")
plt.savefig("../../fig/slide_spot/slide_spot_mosaic.png", dpi=300)

## filter, remove the interferential spectrum
feta_up = cp.arange(-Na_up/2, Na_up/2) * PRF / Na
f_tau = cp.fft.fftshift(cp.arange(-Nr/2, Nr/2) * Fr / Nr)
mat_ftau_up, mat_feta_up = cp.meshgrid(f_tau, feta_up)

H2 = cp.abs(mat_feta_up - 2*vr*(mat_ftau_up)*cp.sin(theta_c)/c)<PRF/2
echo_mosaic_filted = echo_mosaic * H2

eta_up = cp.arange(-Na_up/2, Na_up/2) / (PRF*copy_cnt)
mat_tau_up, mat_eta_up = cp.meshgrid(tau_spot, eta_up)

H2 = cp.exp(-1j * cp.pi * k_rot * mat_eta_up**2)
# H2 = 1
echo = echo_mosaic_filted*H2
echo_ftau_feta = cp.fft.fft(echo, axis=0)

plt.figure(5)
plt.subplot(1, 2, 1)
plt.imshow(cp.abs(cp.fft.fftshift(S_ftau_feta_spot, axes=1)).get(), aspect='auto')
plt.title("no preprocess")
plt.subplot(1, 2, 2)
plt.imshow(cp.abs(cp.fft.fftshift(echo_ftau_feta, axes=1)).get(), aspect='auto')
plt.title("mosaic flitered result")
plt.savefig("../../fig/slide_spot/slide_spot_mosaic_filtered.png", dpi=300)

## RFM
R_ref = R0
H3 = cp.exp((4j*cp.pi*R_ref/c)*cp.sqrt((f+mat_ftau_up)**2 - c**2 * mat_feta_up**2 / (4*vr**2*R_ref)) + 1j*cp.pi*mat_ftau_up**2/Kr - 1j*cp.pi*mat_feta_up**2/k_rot)
echo_ftau_feta = echo_ftau_feta * H3

## modified stolt mapping
[mat_f_tau_stolt, mat_f_eta_stolt] = cp.meshgrid(f_tau, feta_up)
map_f_tau = cp.sqrt((f+mat_f_tau_stolt)**2-c**2*mat_f_eta_stolt**2/(4*vr**2))-cp.sqrt(f**2 - c**2*mat_f_eta_stolt**2/(4*vr**2))
# map_f_tau = cp.sqrt((f+mat_f_tau_stolt)**2-c**2*mat_f_eta_stolt**2/(4*vr**2))-f
mat_map_f_pos = (map_f_tau)/(Fr/Nr) #频率转index
mat_map_f_pos = (mat_map_f_pos<0)*Nr + mat_map_f_pos
mat_map_f_int = cp.floor(mat_map_f_pos)
mat_map_f_remain = mat_map_f_pos-mat_map_f_int

## sinc interpolation kernel length, used by stolt mapping
sinc_N = 8

# 定义并行计算的核函数
kernel_code = '''
extern "C" 
#define M_PI 3.14159265358979323846
__global__ void sinc_interpolation(
    const double* echo_ftau_feta,
    const double* Ext_map_f_int,
    const double* Ext_map_f_remain,
    double* echo_ftau_feta_stolt,
    double* check_matrix,
    int Na, int Nr, int sinc_N) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < Na && j < Nr) {
        int map_f_int = Ext_map_f_int[i * Nr + j];
        double map_f_remain = Ext_map_f_remain[i * Nr + j];
        double predict_value = 0;
        check_matrix[i * Nr + j] = sinc_N/2;
        if(map_f_remain < 1e-6) {
            echo_ftau_feta_stolt[i * Nr + j] = echo_ftau_feta[i * Nr + map_f_int];
            return;
        }
        for (int m = 0; m < sinc_N; ++m) {
            double sinc_x = map_f_remain - (m - sinc_N/2);
            double sinc_y = sin(M_PI * sinc_x) / (M_PI * sinc_x);
            int index = map_f_int + m - sinc_N/2;
            if (index >= Nr) {
                predict_value += echo_ftau_feta[i * Nr + (Nr - 1)] * sinc_y;
            } else if (index < 0) {
                predict_value += echo_ftau_feta[i * Nr] * sinc_y;
            } else {
                predict_value += echo_ftau_feta[i * Nr + index] * sinc_y;
            }
        }
        echo_ftau_feta_stolt[i * Nr + j] = predict_value;
    }
}
'''

# 编译核函数
module = cp.RawModule(code=kernel_code)
sinc_interpolation = module.get_function('sinc_interpolation')
echo_ftau_feta = cp.ascontiguousarray(echo_ftau_feta)
# 初始化数据
echo_ftau_feta_stolt_real = cp.zeros((Na_up, Nr), dtype=cp.double)
echo_ftau_feta_stolt_imag = cp.zeros((Na_up, Nr), dtype=cp.double)
echo_ftau_feta_real = cp.real(echo_ftau_feta).astype(cp.double)
echo_ftau_feta_imag = cp.imag(echo_ftau_feta).astype(cp.double)
check_matrix = cp.zeros((Na_up, Nr), dtype=cp.double)

# 设置线程和块的维度
threads_per_block = (16, 16)
blocks_per_grid = (int(cp.ceil(Na_up / threads_per_block[0])), int(cp.ceil(Nr / threads_per_block[1])))

# 调用核函数
sinc_interpolation(
    (blocks_per_grid[0], blocks_per_grid[1]), (threads_per_block[0], threads_per_block[1]),
    (echo_ftau_feta_real, mat_map_f_int, mat_map_f_remain, echo_ftau_feta_stolt_real, check_matrix, Na_up, Nr, sinc_N)
)

sinc_interpolation(
    (blocks_per_grid[0], blocks_per_grid[1]), (threads_per_block[0], threads_per_block[1]),
    (echo_ftau_feta_imag, mat_map_f_int, mat_map_f_remain, echo_ftau_feta_stolt_imag, check_matrix, Na_up, Nr, sinc_N)
)

echo_ftau_feta_stolt = echo_ftau_feta_stolt_real + 1j * echo_ftau_feta_stolt_imag

echo_tau_feta = cp.fft.ifft(echo_ftau_feta_stolt, axis=1)
mat_R = mat_tau_up*c/2
H4 = cp.exp((4j*cp.pi*(mat_R - R_ref)/c) * cp.sqrt(f**2 - c**2 * mat_f_eta_stolt**2 / (4 * vr**2))) * cp.exp(2j*cp.pi*mat_f_eta_stolt*eta_c_spot - 2j*cp.pi*mat_f_eta_stolt*feta_c/k_rot)
echo_tau_feta = echo_tau_feta * H4
echo = cp.fft.ifft(echo_tau_feta, axis=0)
plt.figure(6)
plt.subplot(1,2,1)
plt.imshow(cp.abs(cp.fft.fftshift(echo_ftau_feta, axes=1)).get(), aspect='auto')
plt.title("before stolt mapping")
plt.subplot(1,2,2)
plt.imshow(cp.abs(cp.fft.fftshift(echo, axes=1)).get(), aspect='auto')
plt.title("after stolt mapping")
plt.savefig("../../fig/slide_spot/slide_spot_stolt.png", dpi=300)