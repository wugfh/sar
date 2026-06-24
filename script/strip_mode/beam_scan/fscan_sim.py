import numpy as np
import cupy as cp
import matplotlib.pyplot as plt 
import sys
sys.path.append(r"../../")

from matplotlib import font_manager
import matplotlib

from fscan import Fscan
from dot_estimate import DotEstimator
from autofocus import AutoFocus
import time
import scipy.interpolate as intp 
import scipy.io as sio

cp.cuda.Device(0).use()

def estimate_rcm(sig, fscan_sim):
    midx = cp.argmax(cp.abs(sig), axis=1)
    ac_max = cp.max(cp.abs(sig), axis=1)
    midx[ac_max<cp.max(ac_max)*0.1] = cp.median(midx)
    midx = midx - cp.mean(midx)
    midx = midx

    dR_true = midx*(fscan_sim.c/(2*fscan_sim.Fs))
    return dR_true.get()

def fscan_simulation():
    fscan_sim = Fscan()

    forward = cp.array(sio.loadmat("./pos.mat")["forward"].flatten())
    right = cp.array(sio.loadmat("./pos.mat")["right"].flatten())
    down = cp.array(sio.loadmat("./pos.mat")["down"].flatten())
    down = (down - cp.mean(down))
    right = (right - cp.mean(right))
    forward = forward - cp.mean(forward)


    fscan_sim.set_Vr(float((forward[-1]-forward[0])/(fscan_sim.Na/fscan_sim.PRF)))

    tau = 2*fscan_sim.R0/fscan_sim.c + cp.arange(-fscan_sim.Nr/2, fscan_sim.Nr/2, 1)*(1/fscan_sim.Fs)
    eta = cp.arange(-fscan_sim.Na/2, fscan_sim.Na/2, 1)*(1/fscan_sim.PRF)  
    eta = eta[:,cp.newaxis]
    tau = tau[cp.newaxis, :]
    
    dR = cp.zeros((fscan_sim.Na, 1))
    ftau = cp.arange(-fscan_sim.Nr/2, fscan_sim.Nr/2, 1)*(fscan_sim.Fs/fscan_sim.Nr)
    R = tau*fscan_sim.c/2
    snr = 50


    error_size = forward.shape[0]
    eta_error = cp.linspace(eta[0],eta[-1], error_size) 
    right = cp.interp(cp.squeeze(eta), cp.squeeze(eta_error), cp.squeeze(right))
    down = cp.interp(cp.squeeze(eta), cp.squeeze(eta_error), cp.squeeze(down))

    down = down - fscan_sim.H
    down = cp.squeeze(down)[:, cp.newaxis]
    right = right[:, cp.newaxis]


    forward = cp.interp(cp.squeeze(eta), cp.squeeze(eta_error), cp.squeeze(forward))
    forward = forward - fscan_sim.Rc*cp.sin(fscan_sim.theta_c)
    eta = eta-fscan_sim.Rc*cp.sin(fscan_sim.theta_c)/fscan_sim.Vr
    forward = eta*fscan_sim.Vr
    down = -fscan_sim.H
    right = 0
    

    echo = fscan_sim.echogen(snr, forward, down, right)
    R_real = []
    R_ideal = []
    for i in range(fscan_sim.points_n):
        R_real_item = cp.sqrt((down)**2 + (right-fscan_sim.points_y[i])**2 + (forward - fscan_sim.points_a[i])**2)
        R_ideal_item = cp.sqrt(fscan_sim.points_r[i]**2 + (forward - fscan_sim.points_a[i])**2)
        R_real.append(R_real_item)
        R_ideal.append(R_ideal_item)
    R_real = cp.array(R_real)
    R_ideal = cp.array(R_ideal)
    R_error = R_real - R_ideal
    R_error = cp.squeeze(R_error)

    print("R_error shape", R_error.shape)

    if R_error.shape[0]<R_error.shape[1]:
        R_error = R_error.T

    afocus = AutoFocus(fscan_sim.Fs, fscan_sim.Tp, fscan_sim.f0, fscan_sim.PRF, fscan_sim.Vr, fscan_sim.B, fscan_sim.feta_c, fscan_sim.R0, fscan_sim.theta_width)
    # echo = echo[:, fscan_sim.Nr/2-fscan_sim.Nr/8:fscan_sim.Nr/2+fscan_sim.Nr/8]


    plt.figure()
    plt.imshow(np.abs(echo.get()), aspect='auto', cmap='jet')
    plt.savefig("../../../fig/dbf/fscan_echo.png", dpi=300)




    data_rc = fscan_sim.focus.range_compression(echo)
    
    plt.figure()
    plt.imshow(np.abs(cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(data_rc))).get()), aspect='auto', cmap='jet')
    plt.savefig("../../../fig/dbf/fscan_echo_fft2.png", dpi=300)


    data_rc = cp.array(data_rc)
    # kaiser_win = cp.kaiser(fscan_sim.Na, beta=30)
    # data_rc = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(data_rc, axes=0), axis=0), axes=0)
    # data_rc = data_rc*cp.tile(kaiser_win[:, cp.newaxis], (1, fscan_sim.Nr))
    # data_rc = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(data_rc, axes=0), axis=0), axes=0)

    # data_rc = afocus.Moco_first(data_rc, right-Y,down-fscan_sim.H,forward,cp.deg2rad(60))

    data_rc = fscan_sim.azimuth_interp(cp.array(data_rc), forward=forward)

    data_pre = data_rc

    rcmc = fscan_sim.focus.erma_rcmc(cp.array(data_rc))

    ac = fscan_sim.focus.erma_ac(cp.array(rcmc))

        
    max_value = cp.abs(ac).max()
    noise_level = cp.percentile(cp.abs(ac), 80)
    snr = 20 * cp.log10(max_value / noise_level)
    print(f"Estimated SNR: {snr:.2f} dB before super resolution")

    kfscan = fscan_sim.estimate_kfscan(ac)
    print("estimated kfscan:{}, real kfscan:{}, error:{}".format(kfscan, fscan_sim.Kfscan, abs(kfscan - fscan_sim.Kfscan)/fscan_sim.Kfscan))
    fscan_sim.Kfscan = kfscan
    ac = fscan_sim.fscan_dramp(cp.array(ac))
    fc = fscan_sim.estimate_fscan_center(ac)
    ac = fscan_sim.fscan_shift(cp.array(ac), fc)

    ac_fft2 = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(ac)))
    plt.figure()
    plt.imshow(cp.abs(ac_fft2).get(), aspect='auto', cmap='jet')
    plt.savefig("../../../fig/dbf/fscan_ac_fft2.png", dpi=300)

    ac = fscan_sim.fscan_super_resolution(cp.array(ac))

    print("estimated fscan center:{}".format(fc/1e6))

    image = ac

        
    max_value = cp.abs(image).max()
    noise_level = cp.percentile(cp.abs(image), 80)
    snr = 20 * cp.log10(max_value / noise_level)
    print(f"Estimated SNR: {snr:.2f} dB after super resolution")
    
    # ac, _ = afocus.compensate_R(cp.array(ac), -40, fscan_sim.theta_width)
    # rcmc = fscan_sim.focus.erma_unac(cp.array(ac))

    # dR_before = estimate_rcm(rcmc[:,1000:1100], fscan_sim)
    # ## test run time of pga
    # start_time = time.time()
    # # image = ac

    # ## the first compensation for residual rcm, control the range of rcm into two or three range bins
    # block_cnt = 6
    # da = eta* fscan_sim.Vr
    # snr =  -15*cp.ones(block_cnt+2)
    # pre_win_len = np.zeros_like(snr)
    # for i in range(8,2,-1):
    #     down_rate = i//2
    #     ac_rechirp = afocus.rechirp(cp.array(ac))
    #     ac_dechirp = afocus.dechirp(cp.array(ac_rechirp))

    #     error_line, win_len = afocus.spga(cp.array(ac_dechirp), block_cnt, snr, 30, 10, method="line", range_win=30)
    #     error_line = cp.array(error_line)

    #     print("error value :", cp.abs(error_line).max())
    #     if cp.abs(error_line).max() < 1e-5:
    #             continue

    #     dR += error_line/(4*cp.pi)*fscan_sim.lambda_

    #     data_rc = data_rc*cp.exp(-1j*cp.array(error_line))
    #     rcmc = fscan_sim.focus.erma_rcmc(cp.array(data_rc))
    #     ac = fscan_sim.focus.erma_ac(cp.array(rcmc))
    #     for j in range(block_cnt):
    #         if pre_win_len[j] == 0:
    #             pre_win_len[j] = win_len[j]
    #             snr[j] -= 2
    #         elif win_len[j] < 100:
    #             snr[j] -= 2
    #     print(snr)

    # ## compensate residual rcm
    # dR_intp = dR
    # data_rc_fftr = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(data_pre, axes=1), axis=1), axes=1)
    # data_rc_fftr = data_rc_fftr*cp.exp(-1j*4*cp.pi*dR_intp*(fscan_sim.f0+ftau)/fscan_sim.c)
    # data_rc = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(data_rc_fftr, axes=1), axis=1), axes=1)
    # # data_rc = fscan_sim.azimuth_interp(cp.array(data_rc), forward=forward)
    # rcmc = fscan_sim.focus.erma_rcmc(cp.array(data_rc))
    # ac = fscan_sim.focus.erma_ac(cp.array(rcmc))
    # snr -= 1

    # ## the second compensation for phase error, and control the range of rcm into one range bin
    # for i in range(5):
    #     ac_rechirp = afocus.rechirp(cp.array(ac))
    #     ac = afocus.dechirp(cp.array(ac_rechirp))
    #     error_line, win_len = afocus.spga(cp.array(ac), block_cnt, snr, 30, 10, method="line", range_win=30)
    #     for j in range(block_cnt):
    #         if pre_win_len[j] == 0:
    #             pre_win_len[j] = win_len[j]
    #         elif win_len[j] < 100:
    #             snr[j] -= 1
    #     error_line = cp.array(error_line)

    #     print("error value :", cp.abs(error_line).max())
    #     if cp.abs(error_line).max() < 1e-5:
    #             continue

    #     dR += (error_line)/(4*cp.pi)*fscan_sim.lambda_

    #     dR_intp = dR

    #     data_rc_fftr = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(data_pre, axes=1), axis=1), axes=1)
    #     data_rc_fftr = data_rc_fftr*cp.exp(-1j*4*cp.pi*dR_intp*(fscan_sim.f0+ftau)/fscan_sim.c)
    #     data_rc = cp.fft.ifftshift(cp.fft.ifft(cp.fft.ifftshift(data_rc_fftr, axes=1), axis=1), axes=1)
    #     # data_rc = fscan_sim.azimuth_interp(cp.array(data_rc), forward=forward)

    #     rcmc = fscan_sim.focus.erma_rcmc(cp.array(data_rc))
    #     ac = fscan_sim.focus.erma_ac(cp.array(rcmc))

    #     print(snr)
    # plt.figure()
    # plt.imshow(np.abs(rcmc.get()), aspect='auto', cmap='jet')
    # plt.savefig("../../../fig/dbf/fscan_rcmc.png", dpi=300)

    # dR_after = estimate_rcm(rcmc[:,1000:1100], fscan_sim)
    # ac = fscan_sim.focus.erma_ac(cp.array(rcmc))
    # image = ac.get()
    # # for i in range(fscan_sim.points_n):
    # #     R_error[:, i]= cp.interp(eta*fscan_sim.Vr, cp.array(forward), R_error[:,i])

    # R_error_mean = cp.mean(R_error, axis=1)

    
    # x = cp.arange(R_error_mean.shape[0])
    # slope, intercept = cp.polyfit(x, R_error_mean, 1)
    # R_error_mean = R_error_mean - (slope*x + intercept)
    # R_error_mean = R_error_mean - R_error_mean[0] +dR_intp[0]
    # dR_intp = cp.squeeze(dR_intp)
    # dR = cp.squeeze(dR)
    # print("shape of R_error_mean,dR", R_error_mean.shape, dR_intp.shape)
    
    # plt.figure()
    # plt.plot(-dR_intp.get(), label="error estimated")
    # plt.plot(R_error_mean.get(), label="true error")
    # plt.plot((R_error_mean+dR_intp).get(), label="residual error")
    # plt.xlabel("azimuth time (s)")
    # plt.ylabel("range error (m)")
    # plt.legend()
    # plt.savefig("../../../fig/dbf/dR_estimate.png", dpi=300)


    # plt.figure()
    # plt.plot(dR_before, label="before compensation")
    # plt.plot(dR_after, label="after compensation")
    # plt.xlabel("azimuth time (s)")
    # plt.ylabel("range error (m)")
    # plt.legend()
    # plt.savefig("../../../fig/dbf/R_error.png", dpi=300)



    # # image,error_mat = afocus.spga(cp.array(image), 1, -40, 30, 10, method="mat", range_win=10)

    # # image_e, _ = afocus.spga(cp.array(image_e), R, 3, -40, 30, 10, method="line", range_win=30)

    # end_time = time.time()
    # print(f"Total execution time: {end_time - start_time:.2f} seconds")


    image_ffta = cp.fft.fftshift(cp.fft.fft(cp.fft.fftshift(image, axes=0), axis=0), axes=0)


    plt.figure()
    plt.imshow(np.abs(image_ffta.get()), aspect='auto', cmap='jet')
    plt.savefig("../../../fig/dbf/fscan_R2_ffta.png", dpi=300)



    image_fft2 = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(image)))
    plt.figure()
    plt.imshow(np.abs(image_fft2.get()), aspect='auto', cmap='jet')
    plt.savefig("../../../fig/dbf/fscan_fft2.png", dpi=300)

    image_show = np.abs(image)/np.max(np.max(np.abs(image)))
    image_show = 20*np.log10(image_show)

    plt.figure()
    plt.imshow(image_show.get(), aspect="auto", cmap='jet', vmin=-40, vmax=0)
    plt.colorbar()
    plt.savefig("../../../fig/dbf/fscan_image.png", dpi=300)

    dot_estimator = DotEstimator(fscan_sim.points_n, fscan_sim.c, fscan_sim.Vr, fscan_sim.PRF, fscan_sim.Fs, "../../../fig/dbf/super_")
    dot_estimator.dot_estimate(image.get(), (int(1/(fscan_sim.Vr/fscan_sim.PRF)), int(1/(fscan_sim.c/(2*fscan_sim.Fs)))), 16)


if __name__ == '__main__':

    fscan_simulation()

