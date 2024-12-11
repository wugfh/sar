import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import fsolve
from scipy.signal import decimate
from multiprocessing import Process, Queue

## generate echo
class SlideSpotSim:
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
    c = 299792458
    def __init__(self):
        self.f = 5.6e9  # 载波频率
        self.La = 6
        self.PRF = 2318
        self.Tr = 4e-6
        self.Br = 150e6
        self.Fr = 200e6
        self.vr = 7200
        self.Rc = 600e3
        self.omega = cp.deg2rad(0.2656)  # 波束旋转速度
        self.theta_c = cp.deg2rad(3)
        self.R0 = self.Rc * cp.cos(self.theta_c)
        self.lambda_ = self.c / self.f
        self.Kr = self.Br / self.Tr
        self.Nr = int(cp.ceil(self.Fr * self.Tr))

        self.A = 1 - self.omega * self.R0 / (self.vr * cp.cos(self.theta_c)**2) # 放缩因子
        self.theta_a = 0.886*self.lambda_/self.La # 波束宽度
        self.Tf = 6
        self.Ta = self.Tf  # 成像区域的时间长度
        self.omega_spot = self.vr*cp.cos(self.theta_c)**2/self.Rc
        print("聚束模式波束旋转速度:", cp.rad2deg(self.omega_spot))

        self.R_rot = self.vr*cp.cos(self.theta_c)**2/self.omega

        ## 景中心参数
        self.k_rot = -2 * self.vr**2 * cp.cos(self.theta_c)**3 / (self.lambda_ * self.R_rot)
        self.ka = -2 * self.vr**2 * cp.cos(self.theta_c)**3 / (self.lambda_ * self.R0)
        self.W_spot = self.A*self.vr*self.Ta

        print("Ta, A, k_rot", self.Ta, self.A, self.k_rot)
        print("W_spot", self.W_spot)

        self.Na_spot = int(cp.ceil(self.PRF * self.Ta))
        self.Na_strip = int(cp.ceil(self.PRF * self.Tf))
        self.eta_c_strip = -(self.Rc) * cp.sin(self.theta_c) / self.vr
        self.Bf = 2*0.886*self.vr*cp.cos(self.theta_c)/self.La ## caused by beamwidth
        self.Bsq = 2*self.vr*self.Br*cp.sin(self.theta_c)/self.c  ## casued by squint 
        self.Bs = cp.abs(self.k_rot)*self.Ta  ## caused by rotation
        self.B_tot = self.Bf+self.Bs+self.Bsq
        self.eta_c_spot = -(self.Rc) * cp.sin(self.theta_c) / (self.vr) # 雷达运动到景中心时间
        self.Tx0 = self.Ta-(self.Bf+cp.abs(self.k_rot)*self.Ta)/cp.abs(self.ka)
        self.Tx1 = self.Ta-(cp.abs(self.k_rot)*self.Ta-self.Bf)/cp.abs(self.ka)
        print("Tx0, Tx1", self.Tx0, self.Tx1)
        print("Bf, Bs, Bsq", self.Bf, self.Bs, self.Bsq)

        ## 对于滑动聚束模式，这个值应该大于A
        print("Lf/(vr*Ta)", self.Rc*self.theta_a/(self.vr*self.Ta))


        self.feta_c = 2 * self.vr * cp.sin(self.theta_c) / self.lambda_

        self.point_n = 5
        self.point_r = self.R0+cp.linspace(-200, 200, self.point_n)
        self.point_y = cp.linspace(-12000, 12000, self.point_n)

        print("strip center time, slide spot center time", self.eta_c_strip, self.eta_c_spot)
        

    def generate_echo(self):
        tau_strip = 2*(self.Rc)/self.c + cp.arange(-self.Nr/2, self.Nr/2) / self.Fr
        tau_spot =  2*self.Rc/self.c + cp.arange(-self.Nr/2, self.Nr/2) / self.Fr 
        eta_strip = self.eta_c_strip + cp.arange(-self.Na_strip/2, self.Na_strip/2) / self.PRF
        eta_spot = self.eta_c_spot + cp.arange(-self.Na_spot/2, self.Na_spot/2) / self.PRF

        S_echo_spot = cp.zeros((self.Na_spot, self.Nr), dtype=cp.complex128)
        S_echo_strip = cp.zeros((self.Na_strip, self.Nr), dtype=cp.complex128)
        
        mat_tau_strip, mat_eta_strip = cp.meshgrid(tau_strip, eta_strip)
        mat_tau_spot, mat_eta_spot = cp.meshgrid(tau_spot, eta_spot)
        for i in range(self.point_n):
            # slide spot
            R0_tar = self.point_r[i]
            R_eta_spot = cp.sqrt(R0_tar**2 + (self.vr*mat_eta_spot - self.point_y[i])**2)
            Wr_spot = (cp.abs(mat_tau_spot - 2 * R_eta_spot / self.c) <= self.Tr / 2)

            ##滑动聚束工作模式的斜视角一般定义为当天线波束中心指向场景中心点时的斜视角，因此此时景中心旋转角度为0
            Wa_spot = cp.sinc(self.La * (cp.arccos(R0_tar / R_eta_spot) - (self.theta_c - self.omega * (mat_eta_spot-self.eta_c_spot))) / self.lambda_)**2 
            # Tspot_tar = 0.886*self.lambda_*R0_tar/(self.A*self.La*self.vr*cp.cos(self.theta_c)**2)
            # Wa_spot =  cp.abs(mat_eta_spot-(self.point_y[i]/(self.vr*self.A) + self.eta_c_spot)) < Tspot_tar/2
            Phase_spot = cp.exp(-4j * cp.pi * self.f * R_eta_spot / self.c) * cp.exp(1j * cp.pi * self.Kr * (mat_tau_spot - 2 * R_eta_spot / self.c)**2)
            S_echo_spot += Wr_spot * Wa_spot * Phase_spot

            # strip
            R_eta_strip = cp.sqrt(R0_tar**2 + (self.vr * mat_eta_strip - self.point_y[i])**2)
            Wr_strip = (cp.abs(mat_tau_strip - 2 * R_eta_strip / self.c) <= self.Tr / 2)

            Wa_strip = cp.sinc(self.La * (cp.arccos(R0_tar / R_eta_strip) - self.theta_c) / self.lambda_)**2
            # Tstrip_tar = 0.886*self.lambda_*R0_tar/(self.La*self.vr*cp.cos(self.theta_c)**2)
            # Wa_strip =  cp.abs(mat_eta_strip-(self.point_y[i]/self.vr + self.eta_c_strip)) < Tstrip_tar/2
            Phase_strip = cp.exp(-4j * cp.pi * self.f * R_eta_strip / self.c) * cp.exp(1j * cp.pi *self.Kr * (mat_tau_strip - 2 * R_eta_strip / self.c)**2)
            S_echo_strip += Wr_strip * Wa_strip * Phase_strip

        return S_echo_spot, S_echo_strip

 
   # azimuth dramping, convolve with the signal simliar to transmit signal
    def deramping(self, S_echo_spot):
        tau_spot =  2*self.Rc/self.c + cp.arange(-self.Nr/2, self.Nr/2) / self.Fr 
        # old sample interval
        self.delta_t1 = 1/self.PRF
        self.delta_t2 = 1/(self.B_tot)
        # new sample interval
        R_tranfer = self.R_rot/cp.cos(self.theta_c)**3
        # new sample count
        self.P0 = self.lambda_*R_tranfer/(2*self.vr**2*self.delta_t1 *self.delta_t2)
        self.P0 = int(cp.round(self.P0))
        self.delta_t2 = self.lambda_*R_tranfer/(2*self.vr**2*self.delta_t1 *self.P0)

        eta_1 = self.eta_c_spot + cp.arange(-self.Na_spot/2, self.Na_spot/2)*(self.delta_t1 )
        mat_tau1 , mat_eta_1 = cp.meshgrid(tau_spot, eta_1) 
        H1 = cp.exp(-1j * cp.pi * self.k_rot * mat_eta_1**2 - 2j*cp.pi*self.feta_c*mat_eta_1)

        print("Na_spot, P0:", self.Na_spot, self.P0)
        # convolve with H1, similar to the specan algorithm.
        # remove the doppler centoid varying 
        echo = S_echo_spot*H1
        temp = cp.zeros((self.P0, self.Nr), dtype=cp.complex128)
        temp[self.P0/2-self.Na_spot/2:self.P0/2+self.Na_spot/2, :] = echo  # center
        # temp[0:self.Na_spot, :] = echo #left
        # temp[self.P0-self.Na_spot:self.P0, :] = echo #right
        echo = temp

        ## upsample to P0
        echo_ftau_eta = cp.fft.fftshift(cp.fft.fft2((echo), (self.P0, self.Nr)))

        ## the output azimuth extension
        self.T1 = 0.886*self.lambda_/(self.La*self.omega)
        print("azimuth time after deramping: ", self.T1, self.P0*self.delta_t2)

        # normal ramping, cannot deal with the backfold caused by squint
        eta_2 = cp.arange(-self.P0/2, self.P0/2)*(self.delta_t2)
        _ , mat_eta_2 = cp.meshgrid(tau_spot, eta_2)
        H2 = cp.exp(-1j*cp.pi*self.k_rot*mat_eta_2**2)
        echo_ftau_eta_normal = echo_ftau_eta*H2
        echo_ftau_feta_normal = (cp.fft.fft(echo_ftau_eta_normal, axis=0))
        return echo_ftau_eta, echo_ftau_feta_normal

    ## azimuth mosaic
    def azimuth_mosaic(self, echo_ftau_eta):
        tau_spot =  2*self.Rc/self.c + cp.arange(-self.Nr/2, self.Nr/2) / self.Fr 
        copy_cnt = int(2*cp.ceil((self.Bf+self.Bsq)/(2*self.PRF)) + 1)
        P0_up = self.P0*copy_cnt
        print("copy count:", copy_cnt)
        echo_mosaic = cp.zeros((P0_up, self.Nr), dtype=cp.complex128)
        for i in range(copy_cnt):
            echo_mosaic[i*self.P0:(i+1)*self.P0, :] = echo_ftau_eta

        ## filter, remove the interferential spectrum
        feta_up =  self.feta_c+(cp.arange(-P0_up/2, P0_up/2) * self.PRF/self.P0)
        f_tau = ((cp.arange(-self.Nr/2, self.Nr/2) * self.Fr / self.Nr))
        mat_ftau_up, mat_feta_up = cp.meshgrid(f_tau, feta_up)

        Hf = cp.abs(mat_feta_up - self.feta_c - 2*self.A*self.vr*(mat_ftau_up)*cp.sin(self.theta_c)/self.c)<self.PRF/2
        Hf = cp.roll(Hf, -(self.Na_spot/2), axis=0)
        echo_mosaic_filted = echo_mosaic * Hf
        echo_mosaic_filted = cp.roll(echo_mosaic_filted, (self.Na_spot/2), axis=0)

        self.P1 = int(cp.ceil(self.P0*(self.Bf+self.Bsq)/self.Bf))
        # self.P1 = self.P0
        # delta_t3 = lambda_*R_tranfer/(2*vr**2*self.delta_t1 *P0)
        echo_ftau_eta = echo_mosaic_filted[P0_up/2-self.P1/2:P0_up/2+self.P1/2, :]
        print("P1", self.P1)
        print("new sample rate", 1/self.delta_t2)

        # convolve with H2, upsampled signal. 
        eta_2 = cp.arange(-self.P1/2, self.P1/2)*(self.delta_t2)
        _ , mat_eta_2 = cp.meshgrid(tau_spot, eta_2)

        H2 = cp.exp(-1j*cp.pi*self.k_rot*mat_eta_2**2)
        echo_ftau_eta = echo_ftau_eta*H2
        echo_ftau_feta = (cp.fft.fft(echo_ftau_eta, axis=0))
        self.T1 = self.delta_t2*self.P1
        print("azimuth time after mosaic: ", self.T1)

        return echo_mosaic, echo_ftau_eta, echo_ftau_feta

    def stolt_interpolation(self, echo_ftau_feta, delta_int, delta_remain, Na, Nr, sinc_N):
        module = cp.RawModule(code=self.kernel_code)
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

    def  wk_focusing(self, echo_ftau_feta, k_rot, eta_c, prf, R_ref):
        ## RFM

        [Na,Nr] = cp.shape(echo_ftau_feta)

        f_tau = ((cp.arange(-Nr/2, Nr/2) * self.Fr / Nr))
        f_eta =  self.feta_c+((cp.arange(-Na/2, Na/2) * prf / Na))
        eta = eta_c + cp.arange(-Na/2, Na/2) / prf
        tau = 2 * cp.sqrt(self.R0**2 + self.vr**2 * eta_c**2) / self.c + cp.arange(-Nr/2, Nr/2) /self.Fr

        mat_tau, mat_eta = cp.meshgrid(tau, eta)
        mat_ftau, mat_feta = cp.meshgrid(f_tau, f_eta)

        H3 = cp.exp((4j*cp.pi*R_ref/self.c)*cp.sqrt((self.f+mat_ftau)**2 - self.c**2 * mat_feta**2 / (4*self.vr**2)) + 1j*cp.pi*mat_ftau**2/self.Kr)
        if k_rot != 0:
            H3 =H3*cp.exp(-1j*cp.pi*mat_feta**2/k_rot)
        
        echo_ftau_feta = echo_ftau_feta * H3

        ## modified stolt mapping
        map_f_tau = cp.sqrt((self.f+mat_ftau)**2-self.c**2*mat_feta**2/(4*self.vr**2))-cp.sqrt(self.f**2-self.c**2*mat_feta**2/(4*self.vr**2))
        # map_f_tau = cp.sqrt((f+mat_ftau)**2-c**2*mat_feta**2/(4*vr**2))-f
        delta = (map_f_tau - mat_ftau)/(self.Fr/Nr) #频率转index
        delta_int = cp.floor(delta).astype(cp.int32)
        delta_remain = delta-delta_int

        ## sinc interpolation kernel length, used by stolt mapping
        sinc_N = 8
        echo_ftau_feta_stolt = self.stolt_interpolation(echo_ftau_feta, delta_int, delta_remain, Na, Nr, sinc_N)
        ## focusing
        ## modified stolt mapping, residual azimuth compress
        # mat_R = mat_tau * self.c / 2
        # echo_tau_feta_stolt = cp.zeros((Na, Nr), dtype=cp.complex128)
        # eta_r_c = mat_R * cp.tan(self.theta_c) / self.vr
        # if k_rot != 0:
        #     H4 = cp.exp(4j*cp.pi*(mat_R-R_ref)/self.c * (cp.sqrt(self.f**2-self.c**2*mat_feta**2/(4*self.vr**2)))) * cp.exp(2j*cp.pi*mat_feta*eta_r_c - 2j*cp.pi*mat_feta*self.feta_c/k_rot)
        #     echo_tau_feta_stolt = cp.fft.ifft((echo_ftau_feta_stolt), axis = 1)
        #     echo_tau_feta_stolt = echo_tau_feta_stolt * H4
        # else: 
        #     H4 = cp.exp(4j*cp.pi*(mat_R-R_ref)/self.c * (cp.sqrt(self.f**2-self.c**2*mat_feta**2/(4*self.vr**2)))) *  cp.exp(2j*cp.pi*mat_feta*eta_r_c)
        #     echo_tau_feta_stolt = cp.fft.ifft((echo_ftau_feta_stolt), axis = 1)
        #     echo_tau_feta_stolt = echo_tau_feta_stolt * H4

        # echo_ftau_feta_stolt = cp.fft.fft(echo_tau_feta_stolt, axis = 1)

        echo_stolt = (cp.fft.ifft2((echo_ftau_feta_stolt)))
        echo_no_stolt = (cp.fft.ifft2((echo_ftau_feta)))
        return echo_stolt, echo_no_stolt


    def postfilter(self, echo_spot):
        delta_f1 = 1/(self.T1) ## equal to 1/(self.delta_t2*self.P1)
        f_eta1 =  self.feta_c+(cp.arange(-self.P1/2, self.P1/2) * delta_f1)
        f_tau = ((cp.arange(-self.Nr/2, self.Nr/2) * self.Fr / self.Nr))
        _, mat_feta1 = cp.meshgrid(f_tau, f_eta1)

        echo_tau_feta = cp.fft.fft(echo_spot, axis=0)

        kx = (self.A-1)*self.ka/self.A
        # convolve with H5
        H5 = cp.exp(1j*cp.pi*mat_feta1**2/kx)
        echo_tau_feta = echo_tau_feta * H5
        echo_tau_feta = cp.fft.ifft(echo_tau_feta, axis=0)

        ## azimuth frequency interval
        delta_f2 = cp.array(2/(self.Tx0+self.Tx1))
        self.P2 = int(cp.ceil(cp.abs((kx/(delta_f1*delta_f2)))))
        delta_f2 = cp.abs(kx/(self.P2*delta_f1))
        print("P2: ", self.P2)
        f_eta2 = (cp.arange(-self.P1/2, self.P1/2) * delta_f2)
        _, mat_feta2 = cp.meshgrid(f_tau, f_eta2)
        
        H6 = cp.exp(1j*cp.pi*mat_feta2**2/kx)
        echo_tau_feta = echo_tau_feta * H6

        echo_spot_post = cp.fft.ifft(echo_tau_feta, axis=0)

        print("azimuth time after postfilter: ", 1/delta_f2)

        # eta = self.eta_c_spot + cp.arange(-self.P1/2, self.P1/2) * self.delta_t2
        # tau = 2 * self.Rc / self.c + cp.arange(-self.Nr/2, self.Nr/2) /self.Fr
        # _, mat_eta = cp.meshgrid(tau, eta)
        # H7 = cp.exp(1j*cp.pi*kx*mat_eta**2)
        # echo_spot_post = echo_spot_post * H7
        return echo_spot_post

def normal_foucs_plot(echo_spot_no_pre, echo_strip):
    plt.figure(6)
    plt.subplot(1, 2, 1)
    plt.contour((np.abs(echo_spot_no_pre)), levels=20)
    plt.title("spot no pre  result")
    plt.subplot(1, 2, 2)
    plt.contour((np.abs(echo_strip)), levels=20)
    plt.title("strip result")
    plt.savefig("../../fig/slide_spot/slide_spot_nopre_result.png", dpi=300)

    plt.figure(7)
    plt.subplot(1, 2, 1)
    plt.imshow(np.abs((np.fft.fft2((echo_spot_no_pre)))), aspect='auto')
    plt.title("spot no pre frequcecy")
    plt.subplot(1, 2, 2)
    plt.imshow(np.abs((np.fft.fft2((echo_strip)))), aspect='auto')
    plt.title("strip  frequency")
    plt.savefig("../../fig/slide_spot/slide_spot_nopre_2D_FFT.png", dpi=300) 

def azimuth_mosaic_plot(echo_mosaic, echo_ftau_eta, echo_ftau_feta):
    plt.figure(5)
    plt.subplot(1, 3, 1)
    plt.imshow((np.abs((echo_mosaic))), aspect='auto')
    plt.title("before filtered")
    plt.subplot(1, 3, 2)
    plt.imshow((np.abs((echo_ftau_eta))), aspect='auto')
    plt.title("after filtered")
    plt.subplot(1, 3, 3)
    plt.imshow((np.abs((echo_ftau_feta))), aspect='auto')
    plt.title("after preprocess")
    plt.savefig("../../fig/slide_spot/slide_spot_mosaic_filtered.png", dpi=300)

def dramping_plot(S_tau_feta_spot, echo_ftau_eta, echo_tau_feta_normal):
    plt.figure(4)
    plt.subplot(1, 3, 1)
    plt.imshow(np.abs(((S_tau_feta_spot))), aspect='auto')
    plt.title("after deramping TF")
    plt.subplot(1, 3, 2)
    plt.imshow(np.abs((echo_ftau_eta)), aspect='auto')
    plt.title("after deramping FF")
    plt.subplot(1, 3, 3)
    plt.imshow(np.abs(((echo_tau_feta_normal))), aspect='auto')
    plt.title("after normal ramping FF")
    plt.savefig("../../fig/slide_spot/slide_spot_preprocess_fft2.png", dpi=300)

def generate_echo_plot(S_echo_spot, S_echo_strip):
    S_tau_feta_strip = np.fft.fftshift(np.fft.fft(S_echo_strip, axis=0), axes=0)
    S_tau_feta_spot = np.fft.fftshift(np.fft.fft(S_echo_spot, axis=0), axes=0)
    plt.figure(2)
    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(S_echo_spot), aspect='auto')
    plt.title("slide spotlight TT")
    plt.subplot(1, 2, 2)
    plt.imshow(np.abs(S_echo_strip), aspect='auto')
    plt.title("strip TT")
    plt.savefig("../../fig/slide_spot/slide_spot_strip.png", dpi=300)

    plt.figure(3)
    plt.subplot(1, 2, 1)
    plt.imshow(np.abs((S_tau_feta_spot)), aspect='auto')
    plt.title("slide spotlight TF")
    plt.subplot(1, 2, 2)
    plt.imshow(np.abs((S_tau_feta_strip)), aspect='auto')
    plt.title("strip TF")
    plt.savefig("../../fig/slide_spot/slide_spot_strip_fft2.png", dpi=300)

def postfilter_plot(echo_spot, echo_postfilter):
    plt.figure(8)
    plt.subplot(1, 2, 1)
    plt.contour(np.abs(echo_spot), levels=20)
    plt.title("slide spot stolt result")
    plt.subplot(1, 2, 2)
    plt.contour(np.abs(echo_postfilter), levels=20)
    plt.title("slide spot postfilter result")
    plt.savefig("../../fig/slide_spot/slide_spot_result.png", dpi=300)

    plt.figure(9)
    plt.subplot(1, 2, 1)
    plt.imshow(np.abs((np.fft.fft(echo_spot, axis=0))), aspect='auto')
    plt.title("slide spot stolt frequency")
    plt.subplot(1, 2, 2)
    plt.imshow(np.abs((np.fft.fft(echo_postfilter, axis=0))), aspect='auto')
    plt.title("slide spot postfilter frequency")
    plt.savefig("../../fig/slide_spot/slide_spot_2D_FFT.png", dpi=300)


def calculate_rho(image):
    pos = cp.argmax(image, axis=0)
    print(cp.shape(pos))

def simulate_slide_spot(qfunc, qargs):
    # 定义参数
    cp.cuda.Device(2).use()
    simulate = SlideSpotSim()

    S_echo_spot, S_echo_strip = simulate.generate_echo()

    qfunc.put(generate_echo_plot)
    qargs.put((S_echo_spot.get(), S_echo_strip.get()))


    echo_strip, _ = simulate.wk_focusing(cp.fft.fftshift(cp.fft.fft2(((S_echo_strip)))), 0, simulate.eta_c_strip, simulate.PRF, simulate.R0)


    delta_t2 = 1/simulate.B_tot
    echo_ftau_eta, echo_ftau_feta_normal = simulate.deramping(S_echo_spot)
    qfunc.put(dramping_plot)
    qargs.put((echo_ftau_eta.get(), cp.fft.fft(echo_ftau_eta, axis=0).get(), (cp.fft.ifft2(echo_ftau_feta_normal)).get()))

    echo_spot_no_moasic, _ = simulate.wk_focusing(echo_ftau_feta_normal, 0, simulate.eta_c_spot, simulate.PRF, simulate.R0)
    qfunc.put(normal_foucs_plot)
    qargs.put((echo_spot_no_moasic.get(), echo_strip.get()))

    echo_mosaic, echo_ftau_eta, echo_ftau_feta = simulate.azimuth_mosaic(echo_ftau_eta)
    qfunc.put(azimuth_mosaic_plot)
    qargs.put((echo_mosaic.get(), echo_ftau_eta.get(), echo_ftau_feta.get()))

    echo_spot, _ = simulate.wk_focusing((echo_ftau_feta), simulate.k_rot, simulate.eta_c_spot, 1/delta_t2, simulate.R0)
    echo_spot_postfilter = simulate.postfilter(echo_spot)
    qfunc.put(postfilter_plot)
    qargs.put((echo_spot.get(), echo_spot_postfilter.get()))
    print("foucsing done")


def plot_sim(qfunc, qargs):
    matplotlib.use('Agg')
    matplotlib.style.use('fast')
    print("plot process start")
    while True:
        args = qargs.get()
        func = qfunc.get()
        if args is None:
            break
        func(*args)
        print(func.__name__, " done")

if __name__ == "__main__":
    qfunc = Queue()
    qargs = Queue()
    plot_process = Process(target=plot_sim, args=(qfunc, qargs))
    plot_process.start()

    simulate_process = Process(target=simulate_slide_spot, args=(qfunc, qargs))
    simulate_process.start()
    simulate_process.join()

    ## terminate the plot process
    qfunc.put(None)
    qargs.put(None)
    plot_process.join()
    exit(0)





