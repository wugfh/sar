import numpy as np
import cupy as cp
import matplotlib.pyplot as plt 
import sys
sys.path.append(r"../../")

from matplotlib import font_manager

from fscan import Fscan
from dot_estimate import DotEstimator
from autofocus import AutoFocus

my_font = font_manager.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")

cp.cuda.Device(1).use()

def dechirp(data, Ka, Fr, PRF):
    Na, Nr = cp.shape(data)
    tau = (cp.linspace(-Nr/2,Nr/2-1,Nr))*(1/Fr)
    eta = (cp.linspace(-Na/2,Na/2-1,Na))*(1/PRF)
    mat_tau, mat_eta = cp.meshgrid(tau, eta) 
    # mat_R0 = mat_tau*self.c/2 + self.R0;  

    data = data*cp.exp(1j*cp.pi*Ka*mat_eta**2)
    data = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(data, axes=0), axis=0), axes=0)
    return data.get()

def spga(sig, ka, afocus: AutoFocus, Fr, PRF):
    [Na, Nr] = sig.shape
    

    bsize = int(2*Na/3)
    block_len = bsize/3
    lmid = np.arange(block_len/2, Na, block_len)
    
    step = 0
    focus_image = cp.zeros((Na, Nr), dtype=cp.complex128)
    for mid in lmid:
        step += 1
        start = np.maximum(0, int(mid - bsize/2))
        end = int(np.minimum(start+bsize, Na))
        print("step:{}, start:{}, end:{}".format(step, start, end))
        W = cp.zeros_like(sig)
        W[start:end, :] = 1
        block = sig*W
        block_dechirp = dechirp(cp.array((block)), ka, Fr, PRF)
        error, rms, windata = afocus.pga_autofocus(cp.array((block_dechirp)), num_iter=30, snr = -30)
        print("RMS error:\r\n", rms)
        error_sum = cp.array(error)

        error_sum = cp.tile(error_sum[:, cp.newaxis], (1, Nr))
        
        block_dechirp_ffta  = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(block_dechirp, axes=0), axis=0), axes=0)
        block_dechirp_ffta = block_dechirp_ffta*cp.exp(-1j*error_sum)
        block_dechirp = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(block_dechirp_ffta, axes=0), axis=0), axes=0)
        block = block_dechirp
        focus_image[mid-block_len//2:mid+block_len//2, :] += block[mid-block_len//2:mid+block_len//2, :]
    sig = focus_image.get()
    return sig,error

def fscan_simulation():
    fscan_sim = Fscan()
    echo = fscan_sim.echogen()
    # echo = echo[:, fscan_sim.Nr/2-fscan_sim.Nr/8:fscan_sim.Nr/2+fscan_sim.Nr/8]

    plt.figure()
    plt.imshow(np.abs((echo.get())), aspect='auto', cmap='jet')
    plt.colorbar()
    plt.savefig("../../../fig/dbf/fscan_echo.png", dpi=300)

    tau = 2*fscan_sim.R0/fscan_sim.c + cp.arange(-fscan_sim.Nr/2, fscan_sim.Nr/2, 1)*(1/fscan_sim.Fs)
    mat_R = cp.tile(tau, (fscan_sim.Na, 1))*fscan_sim.c/2

    # image = fscan_sim.focus.wk_focus(echo, fscan_sim.R0).get()
    data_rc = fscan_sim.focus.range_compression(echo)
    print("Range compression done")
    rcmc = fscan_sim.focus.rd_rcmc(data_rc)
    # print("RCMC done")
    afocus = AutoFocus(fscan_sim.Fs, fscan_sim.Tp, fscan_sim.f0, fscan_sim.PRF, fscan_sim.Vr, fscan_sim.B, fscan_sim.feta_c, fscan_sim.R0)
    Ka = 2*fscan_sim.Vr**2*cp.cos(fscan_sim.theta_c)**3/(fscan_sim.lambda_*mat_R)
    image = spga(rcmc, Ka, afocus, fscan_sim.Fs, fscan_sim.PRF)[0]
    # image = fscan_sim.fscan_rd_ac_focus(rcmc)
    image_show = np.abs(image)/np.max(np.max(np.abs(image)))
    image_show = 20*np.log10(image_show)

    image_fft = cp.abs(cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(cp.array(image))))).get()

    plt.figure()
    plt.imshow(((image_fft)), aspect='auto', cmap='jet')
    plt.colorbar()

    plt.xlabel("Range Frequency(MHz)")
    plt.ylabel("Azimuth Frequency(Hz)")
    plt.savefig("../../../fig/dbf/fscan_echo_fft.png", dpi=300)


    plt.figure()
    plt.imshow(image_show, aspect="auto", cmap='jet', vmin=-40, vmax=0)
    plt.colorbar()
    plt.savefig("../../../fig/dbf/fscan_image.png", dpi=300)

    dot_estimator = DotEstimator(5, fscan_sim.c, fscan_sim.Vr, fscan_sim.PRF, fscan_sim.Fs, "../../../fig/dbf/")
    dot_estimator.dot_estimate((image), (int(1.5/(fscan_sim.Vr/fscan_sim.PRF)), int(3/(fscan_sim.c/(2*fscan_sim.Fs)))), 16)


if __name__ == '__main__':
    # fscan_estimate()
    # fscan_carrier_estimate()
    # unsuitable()
    # fscan_ant_d_estimate()
    # fscan_ant_f_estimate()
    fscan_simulation()
    # dbf_simulation()
