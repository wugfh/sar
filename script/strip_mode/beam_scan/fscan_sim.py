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
    mat_tau, mat_eta = cp.meshgrid(tau, eta)
    mat_R = mat_tau*fscan_sim.c/2

    R_error =0.002*mat_eta**4 + 0.01*mat_eta**3 + 0.05*mat_eta**2 + 0.2*mat_eta
    echo = fscan_sim.echogen(R_error,20)
    afocus = AutoFocus(fscan_sim.Fs, fscan_sim.Tp, fscan_sim.f0, fscan_sim.PRF, fscan_sim.Vr, fscan_sim.B*fscan_sim.theta_width/fscan_sim.scan_width, fscan_sim.feta_c, fscan_sim.R0)
    # echo = echo[:, fscan_sim.Nr/2-fscan_sim.Nr/8:fscan_sim.Nr/2+fscan_sim.Nr/8]

    plt.figure()
    plt.imshow(np.abs((echo.get())), aspect='auto', cmap='jet')
    plt.colorbar()
    plt.savefig("../../../fig/dbf/fscan_echo.png", dpi=300)

    data_rc = fscan_sim.focus.range_compression(echo)
    # ac = fscan_sim.focus.wk_focus(data_rc, fscan_sim.R0).get()
    rcmc = fscan_sim.focus.rd_rcmc(data_rc)
    ac = fscan_sim.fscan_rd_ac_focus(rcmc)
    # image = ac
    print("RCMC done")
    afocus = AutoFocus(fscan_sim.Fs, fscan_sim.Tp, fscan_sim.f0, fscan_sim.PRF, fscan_sim.Vr, fscan_sim.B, fscan_sim.feta_c, fscan_sim.R0)

    ### test run time of pga
    start_time = time.time()
    image,error = afocus.spga(ac, mat_R, 1, -40, 30, 10)

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

    error = np.concatenate(error, axis=0)
    plt.figure()
    plt.imshow(error, aspect='auto', cmap='jet')
    plt.colorbar(label="pga error")
    plt.xlabel("Range lines/block")
    plt.ylabel("Azimuth lines/block")
    plt.savefig("../../../fig/dbf/pga_range_error.png", dpi=300)

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
    dot_estimator.dot_estimate((image), (int(3/(fscan_sim.Vr/fscan_sim.PRF)), int(3/(fscan_sim.c/(2*fscan_sim.Fs)))), 16)


if __name__ == '__main__':

    fscan_simulation()

