import numpy as np
import cupy as cp
import matplotlib.pyplot as plt 
import sys
sys.path.append(r"../../")

from matplotlib import font_manager

from fscan import Fscan
from dot_estimate import DotEstimator
from autofocus import AutoFocus
import time
import scipy.io as sio

my_font = font_manager.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


cp.cuda.Device(0).use()

def fscan_simulation():
    fscan_sim = Fscan()
    tau = 2*fscan_sim.R0/fscan_sim.c + cp.arange(-fscan_sim.Nr/2, fscan_sim.Nr/2, 1)*(1/fscan_sim.Fs)
    eta_c = -fscan_sim.Rc*cp.sin(fscan_sim.theta_c)/fscan_sim.Vr
    eta = eta_c + cp.arange(-fscan_sim.Na/2, fscan_sim.Na/2, 1)*(1/fscan_sim.PRF)  
    _, mat_eta = cp.meshgrid(tau, eta)
    R = tau*fscan_sim.c/2
    snr = 20

    forward = cp.array(sio.loadmat("./pos.mat")["forward"].flatten())
    right = cp.array(sio.loadmat("./pos.mat")["right"].flatten())
    down = cp.array(sio.loadmat("./pos.mat")["down"].flatten())
    down = down - cp.mean(down)
    right = right - cp.mean(right)

    error_size = forward.shape[0]
    eta_error = eta_c + cp.arange(-error_size/2, error_size/2, 1)*(1/fscan_sim.PRF) 
    right = cp.interp(eta, eta_error, right)
    down = cp.interp(eta, eta_error, down)
    Y = cp.sqrt(fscan_sim.R0**2-fscan_sim.H**2)

    forward = cp.interp(eta, eta_error, forward)

    R_error = cp.sqrt((right-Y)**2 + (down-fscan_sim.H)**2 + forward**2)-cp.sqrt(forward**2+fscan_sim.R0**2)

    mat_R_error = cp.tile(R_error[:, cp.newaxis], (1, fscan_sim.Nr))

    echo = fscan_sim.echogen(mat_R_error,snr, forward)
    afocus = AutoFocus(fscan_sim.Fs, fscan_sim.Tp, fscan_sim.f0, fscan_sim.PRF, fscan_sim.Vr, fscan_sim.B, fscan_sim.feta_c, fscan_sim.R0, fscan_sim.theta_width)
    # echo = echo[:, fscan_sim.Nr/2-fscan_sim.Nr/8:fscan_sim.Nr/2+fscan_sim.Nr/8]

    plt.figure()
    plt.imshow(np.abs(echo.get()), aspect='auto', cmap='jet')
    plt.savefig("../../../fig/dbf/fscan_echo.png", dpi=300)

    data_rc = fscan_sim.focus.range_compression(echo)
    data_rc = cp.array(data_rc)

    # data_rc = afocus.Moco_first(data_rc, right-Y,down-fscan_sim.H,forward,cp.deg2rad(60))
    data_rc = fscan_sim.azimuth_interp(cp.array(data_rc), forward=forward)
    
    rcmc = fscan_sim.focus.erma_rcmc(cp.array(data_rc))

    ac = fscan_sim.focus.erma_ac(rcmc)

    # ## align point compensation
    # da = cp.arange(-fscan_sim.Na/2, fscan_sim.Na/2, 1)*(fscan_sim.Vr/fscan_sim.PRF)
    # Kr = (cp.arange(-fscan_sim.Nr/2, fscan_sim.Nr/2, 1)*(fscan_sim.Fs/fscan_sim.Nr)+fscan_sim.f0)/fscan_sim.c
    # mat_kr,mat_da = cp.meshgrid(Kr, da)
    # kx_shift = mat_kr/(2*fscan_sim.R0)*mat_da
    # sig_fftr = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(ac, axes=1), axis=1), axes=1)
    # sig_fftr_align = sig_fftr * cp.exp(-1j*kx_shift*mat_da)

    # ac_fft2_before = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(ac))).get()

    # # ac = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(sig_fftr_align, axes=1), axis=1), axes=1).get()

    # ac_fft2_after = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(ac))).get()


    ### test run time of pga
    start_time = time.time()
    # image = ac
    image,error_line = afocus.spga(cp.array(ac), 6, -40, 30, 10, method="line", range_win=30)



    # image,error_mat = afocus.spga(cp.array(image), 1, -40, 30, 10, method="mat", range_win=10)

    # image_e, _ = afocus.spga(cp.array(image_e), R, 3, -40, 30, 10, method="line", range_win=30)

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")



    image_ffta = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(image, axes=0), axis=0), axes=0)


    plt.figure()
    plt.imshow(np.abs(image_ffta.get()), aspect='auto', cmap='jet')
    plt.savefig("../../../fig/dbf/fscan_R2_ffta.png", dpi=300)

    
    ac_ffta = image_ffta

    ac_ffta = ac_ffta[:,1250:1750]
    midx = cp.argmax(cp.abs(ac_ffta), axis=1)
    ac_max = cp.max(cp.abs(ac_ffta), axis=1)
    midx[ac_max<cp.max(ac_max)*0.1] = cp.median(midx)
    midx = midx - cp.mean(midx)
    midx = midx.get()



    dR = midx*(fscan_sim.c/(2*fscan_sim.Fs))

    plt.figure()
    plt.plot(eta.get(), dR, label="estimated dR")
    plt.plot(eta.get(), R_error.get(), label="true dR")
    plt.xlabel("eta (s)")
    plt.ylabel("dR (m)")
    plt.legend()
    plt.savefig("../../../fig/dbf/R_error.png", dpi=300)



    plt.figure()
    plt.imshow(np.abs(rcmc.get()), aspect='auto', cmap='jet')
    plt.savefig("../../../fig/dbf/fscan_R2.png", dpi=300)

    image_show = np.abs(image)/np.max(np.max(np.abs(image)))
    image_show = 20*np.log10(image_show)

    plt.figure()
    plt.imshow(image_show, aspect="auto", cmap='jet', vmin=-40, vmax=0)
    plt.colorbar()
    plt.savefig("../../../fig/dbf/fscan_image.png", dpi=300)

    dot_estimator = DotEstimator(fscan_sim.points_n, fscan_sim.c, fscan_sim.Vr, fscan_sim.PRF, fscan_sim.Fs, "../../../fig/dbf/")
    dot_estimator.dot_estimate(image, (int(1/(fscan_sim.Vr/fscan_sim.PRF)), int(1/(fscan_sim.c/(2*fscan_sim.Fs)))), 16)


if __name__ == '__main__':

    fscan_simulation()

