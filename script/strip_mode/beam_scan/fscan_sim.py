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

my_font = font_manager.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


cp.cuda.Device(0).use()

def fscan_simulation():
    fscan_sim = Fscan()
    tau = 2*fscan_sim.R0/fscan_sim.c + cp.arange(-fscan_sim.Nr/2, fscan_sim.Nr/2, 1)*(1/fscan_sim.Fs)
    eta_c = -fscan_sim.Rc*cp.sin(fscan_sim.theta_c)/fscan_sim.Vr
    eta = eta_c + cp.arange(-fscan_sim.Na/2, fscan_sim.Na/2, 1)*(1/fscan_sim.PRF)  
    _, mat_eta = cp.meshgrid(tau, eta)
    R = tau*fscan_sim.c/2
    snr = 200

    R_error =0.002*mat_eta**4 + 0.01*mat_eta**3 + 0.05*mat_eta**2 + 0.2*mat_eta
    echo = fscan_sim.echogen(R_error,snr)
    afocus = AutoFocus(fscan_sim.Fs, fscan_sim.Tp, fscan_sim.f0, fscan_sim.PRF, fscan_sim.Vr, fscan_sim.B*fscan_sim.theta_width/fscan_sim.scan_width, fscan_sim.feta_c, fscan_sim.R0)
    # echo = echo[:, fscan_sim.Nr/2-fscan_sim.Nr/8:fscan_sim.Nr/2+fscan_sim.Nr/8]


    data_rc = fscan_sim.focus.range_compression(echo)
    # ac = fscan_sim.focus.wk_focus(data_rc, fscan_sim.R0).get()

    rcmc = fscan_sim.focus.rd_rcmc(data_rc)

    ac = fscan_sim.focus.rd_ac(rcmc)
    image_error = ac.get()
    print("RCMC done")

    echo = fscan_sim.echogen(0,snr)
    data_rc = fscan_sim.focus.range_compression(echo)
    # ac = fscan_sim.focus.wk_focus(data_rc, fscan_sim.R0).get()

    rcmc = fscan_sim.focus.rd_rcmc(data_rc)

    ac = fscan_sim.focus.rd_ac(rcmc)
    image = ac.get()

    echo = fscan_sim.echogen(R_error,snr)
    data_rc = fscan_sim.focus.range_compression(echo)
    # ac = fscan_sim.focus.wk_focus(data_rc, fscan_sim.R0).get()

    rcmc = fscan_sim.focus.rd_rcmc(data_rc)

    ac = fscan_sim.focus.rd_ac(rcmc)
    image_e = ac.get()

    afocus = AutoFocus(fscan_sim.Fs, fscan_sim.Tp, fscan_sim.f0, fscan_sim.PRF, fscan_sim.Vr, fscan_sim.B, fscan_sim.feta_c, fscan_sim.R0)

    ### test run time of pga
    start_time = time.time()
    # image_error,error_line = afocus.spga(cp.array(image_error), R, 3, -40, 30, 10, method="line", range_win=30)
    # image_error,error_mat = afocus.spga(cp.array(image_error), R, 3, -40, 30, 10, method="mat", range_win=10)

    # image_e, _ = afocus.spga(cp.array(image_e), R, 3, -40, 30, 10, method="line", range_win=30)

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

    # error = (error_line)
    # plt.figure()
    # plt.imshow(error, aspect='auto', cmap='jet')
    # plt.colorbar(label="pga error")
    # plt.xlabel("Range lines/block")
    # plt.ylabel("Azimuth lines/block")
    # plt.savefig("../../../fig/dbf/pga_range_error.png", dpi=300)

    image_show = np.abs(image)/np.max(np.max(np.abs(image_error)))
    image_show = 20*np.log10(image_show)

    image_fft = (np.fft.fftshift(np.fft.fft(np.fft.fftshift((image), axes=0), axis=0), axes=0))
    image_error_fft = (np.fft.fftshift(np.fft.fft(np.fft.fftshift((image_error), axes=0), axis=0), axes=0))
    image_e_fft = (np.fft.fftshift(np.fft.fft(np.fft.fftshift((image_e), axes=0), axis=0), axes=0))

    range_width = int(3/(fscan_sim.c/(2*fscan_sim.Fs)))
    image_select_error = image_error_fft.copy()
    image_select = image_e_fft.copy()
    plt.figure()
    for i in range(fscan_sim.points_n):
        midx =  np.unravel_index(np.argmax(np.abs(image_select_error)), image_select_error.shape)
        val = image_select_error[:, midx[1]]
        threshold = np.max(np.abs(val))*0.3
        val = val[np.abs(val)>threshold]

        phase = np.unwrap(np.diff(np.angle(val)))
        phase = phase-np.mean(phase)
        # plt.subplot(2,1,1)
        plt.plot(phase, label=f"point {i+1}")
        image_select_error[:, midx[1]-range_width:midx[1]+range_width] = 0

        # midx = np.unravel_index(np.argmax(np.abs(image_select)), image_select.shape)
        # val = image_select[:, midx[1]]
        # threshold = np.max(np.abs(val))*0.3
        # val = val[np.abs(val)>threshold]
        # phase = np.unwrap(np.diff(np.angle(val)))
        # phase = phase-np.mean(phase)
        # plt.subplot(2,1,2)
        # plt.plot(phase, label=f"point {i+1}")
        # image_select[:, midx[1]-range_width:midx[1]+range_width] = 0    
    # plt.subplot(2,1,1)
    plt.grid()
    plt.xlabel("Azimuth ")
    plt.ylabel("Phase gradient(rad)")
    plt.ylim(-0.2, 0.2)
    plt.legend()
    plt.tight_layout()
    # plt.subplot(2,1,2)
    # plt.grid()
    # plt.xlabel("Azimuth")
    # plt.ylabel("Phase error gradient(rad)")
    # plt.ylim(-0.2, 0.2)
    # plt.legend()
    # plt.tight_layout()
    plt.savefig("../../../fig/dbf/error_range.png", dpi=1000)

    plt.figure()
    plt.imshow(image_show, aspect="auto", cmap='jet', vmin=-40, vmax=0)
    plt.colorbar()
    plt.savefig("../../../fig/dbf/fscan_image.png", dpi=300)

    dot_estimator = DotEstimator(fscan_sim.points_n, fscan_sim.c, fscan_sim.Vr, fscan_sim.PRF, fscan_sim.Fs, "../../../fig/dbf/")
    dot_estimator.dot_estimate(image, (int(3/(fscan_sim.Vr/fscan_sim.PRF)), int(3/(fscan_sim.c/(2*fscan_sim.Fs)))), 16)


if __name__ == '__main__':

    fscan_simulation()

