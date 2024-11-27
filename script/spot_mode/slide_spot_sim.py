import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

cp.cuda.Device(3).use()
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
Nr = int(cp.ceil(Fr * Tr))

A = 1 - omega * R0 / (vr * cp.cos(theta_c)**2) # 放缩因子
Ta =  0.886*Rc*lambda_/(A*La*vr*cp.cos(theta_c))
# Ta = 6
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

point_n = 5
point_r = Rc+cp.array([-200, -100, 0, 100, 200])
point_y = cp.array([-2000, -1000, 0, 1000, 2000])


S_echo_spot = cp.zeros((Na, Nr), dtype=cp.complex128)
S_echo_strip = cp.zeros((Na, Nr), dtype=cp.complex128)


## generate echo
for i in range(point_n):
    # slide spot
    R0_tar = point_r[i]*cp.cos(theta_c)
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
k_rot = 2 * vr**2 * cp.cos(theta_c)**3 / (lambda_ * R_rot)
ka = 2 * vr**2 * cp.cos(theta_c)**3 / (lambda_ * R0)

# convolve with H1, similar to the specan algorithm.
H1 = cp.exp(-1j * cp.pi * k_rot * mat_eta_spot**2)
echo = S_echo_spot * H1
echo_tau_eta = (cp.fft.fft(echo, Na, axis=0))


echo_ftau_feta = cp.fft.fft2(echo)

plt.figure(4)
plt.subplot(1, 2, 1)
plt.imshow(cp.abs(cp.fft.fftshift(S_ftau_feta_spot)).get(), aspect='auto')
plt.title("before deramping")
plt.subplot(1, 2, 2)
plt.imshow(cp.abs(cp.fft.fftshift(echo_ftau_feta)).get(), aspect='auto')
plt.title("after deramping")
plt.savefig("../../fig/slide_spot/slide_spot_preprocess_fft2.png", dpi=300)

# ## azimuth mosaic
copy_cnt = int(2*cp.ceil((Bf+Bsq)/(2*PRF)) + 1)
Na_up = Na*copy_cnt
print("copy count:", copy_cnt)
echo_mosaic = cp.zeros((Na_up, Nr), dtype=cp.complex128)
for i in range(copy_cnt):
    echo_mosaic[i*Na:(i+1)*Na, :] = echo_ftau_feta

## filter, remove the interferential spectrum
feta_up =  (cp.arange(-Na_up/2, Na_up/2) * PRF / Na)
f_tau = ((cp.arange(-Nr/2, Nr/2) * Fr / Nr))
mat_ftau_up, mat_feta_up = cp.meshgrid(f_tau, feta_up)

Hf = cp.abs(mat_feta_up - 2*(mat_ftau_up+f)*cp.sin(theta_c)/c)<PRF/2
row_max = cp.max(Hf, axis=1)

echo_mosaic_filted = echo_mosaic * Hf

## upsample the signal, and ramping back

eta_up = cp.arange(-Na_up/2, Na_up/2) / (PRF*(Na_up/Na))
mat_tau_up, mat_eta_up = cp.meshgrid(tau_spot, eta_up)

echo_ftau_eta = cp.fft.ifft(echo_mosaic_filted, Na_up, axis=0)

# convolve with H2, upsampled signal. 
H2 = cp.exp(-1j * cp.pi * k_rot * mat_eta_up**2 )
echo_ftau_eta = echo_ftau_eta*H2
echo_ftau_eta = cp.fft.fft(echo_ftau_eta, axis=0)

echo_ftau_feta = cp.fft.fft2(echo_ftau_eta)

plt.figure(5)
plt.subplot(1, 3, 1)
plt.imshow((cp.abs((echo_mosaic))).get(), aspect='auto')
plt.title("before filtered")
plt.subplot(1, 3, 2)
plt.imshow((cp.abs((echo_mosaic_filted))).get(), aspect='auto')
plt.title("after filtered")
plt.subplot(1, 3, 3)
plt.imshow((cp.abs(cp.fft.fftshift(echo_ftau_feta))).get(), aspect='auto')
plt.title("after preprocess")
plt.savefig("../../fig/slide_spot/slide_spot_mosaic_filtered.png", dpi=300)


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
                predict_value += 0;
            } else if (index < 0) {
                predict_value += 0;
            } else {
                predict_value += echo_ftau_feta[i * Nr + index] * sinc_y;
            }
        }
        echo_ftau_feta_stolt[i * Nr + j] = predict_value/sum_sinc;
    }
}
'''
R_ref = R0

def stolt_interpolation(echo_ftau_feta, delta_int, delta_remain, Na, Nr, sinc_N):
    module = cp.RawModule(code=kernel_code)
    sinc_interpolation = module.get_function('sinc_interpolation')
    echo_ftau_feta = cp.ascontiguousarray(echo_ftau_feta)
    # 初始化数据
    echo_ftau_feta_stolt_real = cp.zeros((Na, Nr), dtype=cp.double)
    echo_ftau_feta_stolt_imag = cp.zeros((Na, Nr), dtype=cp.double)
    echo_ftau_feta_real = cp.real(echo_ftau_feta).astype(cp.double)
    echo_ftau_feta_imag = cp.imag(echo_ftau_feta).astype(cp.double)

    # 设置线程和块的维度
    threads_per_block = (16, 16)
    blocks_per_grid = (int(cp.ceil(Na / threads_per_block[0])), int(cp.ceil(Nr / threads_per_block[1])))

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

def  wk_focusing(echo, k_rot, eta_c):
    ## RFM

    [Na,Nr] = cp.shape(echo)

    f_tau = cp.fft.fftshift((cp.arange(-Nr/2, Nr/2) * Fr / Nr))
    f_eta = feta_c + cp.fft.fftshift((cp.arange(-Na/2, Na/2) * PRF / Na))
    eta_strip = eta_c_strip + cp.arange(-Na/2, Na/2) / PRF
    tau_strip = 2 * cp.sqrt(R0**2 + vr**2 * eta_c_strip**2) / c + cp.arange(-Nr/2, Nr/2) / Fr

    mat_tau, mat_eta = cp.meshgrid(tau_strip, eta_strip)
    mat_ftau, mat_feta = cp.meshgrid(f_tau, f_eta)

    echo_ftau_feta = cp.fft.fft2(echo)
    H3_strip = cp.exp((4j*cp.pi*R_ref/c)*cp.sqrt((f+mat_ftau)**2 - c**2 * mat_feta**2 / (4*vr**2)) + 1j*cp.pi*mat_ftau**2/Kr)
    echo_ftau_feta = echo_ftau_feta * H3_strip

    ## modified stolt mapping
    map_f_tau = cp.sqrt((f+mat_ftau)**2-c**2*mat_feta**2/(4*vr**2))-cp.sqrt(f**2-c**2*mat_feta**2/(4*vr**2))
    # map_f_tau = cp.sqrt((f+mat_ftau)**2-c**2*mat_feta**2/(4*vr**2))-f
    delta = (map_f_tau - mat_ftau)/(Fr/Nr) #频率转index
    delta_int = cp.floor(delta).astype(cp.int32)
    delta_remain = delta-delta_int

    print("delta max, min", cp.max(cp.max(delta)), cp.min(cp.min(delta)))

    ## sinc interpolation kernel length, used by stolt mapping
    sinc_N = 8
    echo_ftau_feta_stolt = stolt_interpolation(echo_ftau_feta, delta_int, delta_remain, Na, Nr, sinc_N)

    ## focusing
    ## modified stolt mapping, residual azimuth compress
    mat_R = mat_tau * c / 2
    # if k_rot != 0:
    #     H4 = cp.exp(4j*cp.pi*(mat_R-R_ref)/c * (cp.sqrt(f**2-c**2*mat_feta**2/(4*vr**2)))) * cp.exp(2j*cp.pi*mat_feta*eta_c - 2j*cp.pi*mat_feta*feta_c/k_rot)
    #     H4 = cp.fft.fft(H4, axis=1)
    #     echo_ftau_feta_stolt = echo_ftau_feta_stolt * H4
    # else: 
    #     H4 = cp.exp(4j*cp.pi*(mat_R-R_ref)/c * (cp.sqrt(f**2-c**2*mat_feta**2/(4*vr**2))))
    #     H4 = cp.fft.fft(H4, axis=1)
    #     echo_ftau_feta_stolt = echo_ftau_feta_stolt * H4

    echo_stolt = (cp.fft.ifft2((echo_ftau_feta_stolt)))
    echo_no_stolt = (cp.fft.ifft2((echo_ftau_feta)))
    return echo_stolt, echo_no_stolt, echo_ftau_feta_stolt, echo_ftau_feta

echo_strip, echo_no_stolt, _, _ = wk_focusing(S_echo_strip, 0, eta_c_strip)

plt.figure(6)
plt.subplot(1, 2, 1)
plt.contour((cp.abs(echo_strip)).get(), levels=20)
plt.title("strip stolt result")
plt.subplot(1, 2, 2)
plt.contour((cp.abs(echo_no_stolt)).get(), levels=20)
plt.title("strip no stolt result")
plt.savefig("../../fig/slide_spot/strip_result.png", dpi=300)

plt.figure(7)
plt.subplot(1, 2, 1)
plt.imshow(cp.abs(cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(echo_strip)))).get(), aspect='auto')
plt.title("strip stolt frequency")
plt.subplot(1, 2, 2)
plt.imshow(cp.abs(cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(echo_no_stolt)))).get(), aspect='auto')
plt.title("strip no stolt frequency")
plt.savefig("../../fig/slide_spot/strip_2D_FFT.png", dpi=300)

echo_spot, echo_no_stolt, echo_spot_ftau_feta, echo_no_stolt_ftau_feta = wk_focusing(cp.fft.ifft2(echo_ftau_feta), k_rot, eta_c_spot)

# echo_tau_feta = cp.fft.fft(echo_spot, axis=0)
# kx = (A-1)*ka/A
# # convolve with H5
# H5 = cp.exp(1j*cp.pi*mat_feta_up**2/kx)
# echo_tau_feta = echo_tau_feta * H5
# echo_tau_feta = cp.fft.fft(echo_tau_feta, axis=0)

plt.figure(8)
plt.subplot(1, 2, 1)
plt.contour(cp.abs(echo_spot).get(), levels=20)
plt.title("slide spot stolt result")
plt.subplot(1, 2, 2)
plt.contour(cp.abs(echo_no_stolt).get(), levels=20)
plt.title("slide spot no stolt result")
plt.savefig("../../fig/slide_spot/slide_spot_result.png", dpi=300)

plt.figure(9)
plt.subplot(1, 2, 1)
plt.imshow(cp.abs(echo_spot_ftau_feta).get(), aspect='auto')
plt.title("slide spot stolt frequency")
plt.subplot(1, 2, 2)
plt.imshow(cp.abs(echo_no_stolt_ftau_feta).get(), aspect='auto')
plt.title("slide spot no stolt frequency")
plt.savefig("../../fig/slide_spot/slide_spot_2D_FFT.png", dpi=300)
