import numpy as np
import cupy as cp
import sys
from tqdm import tqdm
sys.path.append(r"./")
from sinc_interpolation import SincInterpolation

class AutoFocus:
    def __init__(self, Fs, Tp, f0, PRF, Vr, B, fc, R0, Kr):                         
        self.Re = 6371.39e3                     #地球半径
        self.c = 299792458                      #光速
        self.Fs = Fs                                     
        self.Tp = Tp                            #脉冲宽度                        
        self.f0 = f0                            #载频                     
        self.PRF = PRF                          #PRF                     
        self.Vr = Vr                            #雷达速度     
        self.B = B                              #信号带宽
        self.fc = fc                            #多普勒中心频率
        self.lambda_= self.c/self.f0
        self.theta_c = cp.arcsin(self.fc*self.lambda_/(2*self.Vr))
        self.R0 = R0
        self.Rc = self.R0/cp.cos(self.theta_c)
        self.Kr = Kr

    def Moco_first(self, echo, right, down, phi):
        """
        Motion compensation.
        
        Parameters:
        echo (numpy array): echo data before range compress.
        
        Returns:
        numpy array: Motion compensated echo data.
        """
        [Na, Nr] = cp.shape(echo)

        f_tau = cp.fft.fftshift(cp.linspace(-Nr/2,Nr/2-1,Nr)*(self.Fs/Nr))
        f_eta = self.fc + (cp.linspace(-Na/2,Na/2-1,Na)*(self.PRF/Na))

        [mat_f_tau, _] = cp.meshgrid(f_tau, f_eta)
        down = down - cp.mean(down)
        right = right - cp.mean(right)
        r_los = down*cp.cos(phi) - right*cp.sin(phi)
        mat_r_los = r_los[:, cp.newaxis] * cp.ones((1, Nr))
        s_rfft = cp.fft.fft(echo, axis=1)
        H_mcl = cp.exp(4j*cp.pi*(mat_f_tau+self.f0)*mat_r_los/self.c)
        s_rfft_mcl = s_rfft * H_mcl
        echo_mcl = cp.fft.ifft(s_rfft_mcl, axis=1)
        return echo_mcl.get()
    
    def pga(self, corrupted_image, num_iter=10, min_winsize = 32, initial_window_ratio=0.5, snr_threshold=10):
        myImg = corrupted_image.copy()
        imgSize = myImg.shape

        # RMS started at an arbitrary value > .1
        RMS = 10

        # This is where the iteration metric is checked
        # while RMS > .1
        for iter in tqdm(range(num_iter)):
            # Initialization
            centeredImg = cp.zeros(imgSize, dtype=cp.complex128)
            phi = cp.zeros(imgSize[1], dtype=cp.complex128)

            # 1: Center brightest points of image
            maxIdx = cp.argmax(myImg, axis=1)
            midpoint = imgSize[1] // 2

            for i in range(imgSize[0]):
                centeredImg[i, :] = cp.roll(myImg[i, :], midpoint - maxIdx[i])

            # 2: Window Image
            centMag = centeredImg * cp.conj(centeredImg)
            Sx = cp.sum(centMag, axis=0)
            Sx_dB = 20 * cp.log10(cp.abs(Sx))
            cutoff = cp.max(Sx_dB) - 10

            W = 0
            WinBool = Sx_dB >= cutoff
            W = cp.sum(WinBool)

            # Two windows have been tested, a normal curve and a square window
            x = cp.arange(len(Sx))
            W = W * 1.5
            window = (x > (midpoint - W / 2)) & (x < (midpoint + W / 2))
            # window = cp.exp(-(x - midpoint) ** 2 / (2 * (W) ** 2))
            window = cp.tile(window, (imgSize[0], 1))
            windowedImg = centeredImg * window

            # 3. Gradient Generation done by 2 methods

            # Minimum Variance
            Gn = cp.fft.ifft(windowedImg, axis=1)
            dGn = cp.fft.ifft(1j * cp.tile(x - midpoint, (imgSize[0], 1)) * windowedImg, axis=1)

            num = cp.sum(cp.imag(cp.conj(Gn) * dGn), axis=0)
            denom = cp.sum(cp.conj(Gn) * Gn, axis=0)

            dPhi = num / denom
            # Maximum Likelihood
            dPhi2 = cp.zeros(len(dPhi), dtype=cp.complex128)
            for k in range(1, len(phi)):
                dPhi2[k] = cp.sum(Gn[:, k] * cp.conj(Gn[:, k - 1]))
                dPhi2[k] = dPhi2[k] / imgSize[0]
                dPhi2[k] = cp.arctan(cp.imag(dPhi2[k]) / cp.real(dPhi2[k])) + cp.pi / 2 * cp.sin(cp.imag(dPhi2[k])) * (1 - cp.sin(cp.real(dPhi2[k])))

            # Integration of phase gradients to find phase offset
            phi = cp.cumsum(dPhi)
            phi2 = cp.cumsum(dPhi2)

            # The Phase error functions found in each method appear to be very similar.
            # The only major is that the Minimum Variance method seems to have a much greater magnitude
            # so I've been scaling it to the size of the Maximum Likelihood estimation for comparison
            phi = cp.max(cp.abs(phi2)) * phi / cp.max(cp.abs(phi))

            # Add Phase difference estimation to current image and update image
            change = cp.exp(-1j * phi2)
            myImg = cp.fft.fft(cp.fft.ifft(myImg, axis=1) * cp.tile(change, (imgSize[0], 1)), axis=1)

            # find RMS value for removed phase. To be used for iteration
            RMS = cp.sqrt(cp.mean(cp.square(phi2)))
        return myImg.get(), RMS.get()