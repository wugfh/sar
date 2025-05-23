import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import scipy.signal as signal   
import sys
import scipy.optimize as optimize
sys.path.append(r"../")
from sinc_interpolation import SincInterpolation
from sar_focus import SAR_Focus
from mpl_toolkits.mplot3d import Axes3D
import logging
import colorlog
from tqdm import tqdm
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor

from matplotlib.font_manager import FontManager
from matplotlib import font_manager

from fscan import Fscan
from dbf_score import DBF_SCORE
from strip_mode import StripMode

my_font = font_manager.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")

cp.cuda.Device(2).use()

def fscan_simulation():
    fscan_sim = Fscan()
    echo = fscan_sim.echogen()
    # echo = echo[:, fscan_sim.Nr/2-fscan_sim.Nr/8:fscan_sim.Nr/2+fscan_sim.Nr/8]
    Be = fscan_sim.range_estimate()

    # plt.figure()
    # plt.imshow(abs((echo)), aspect='auto', cmap='jet')
    # plt.colorbar()
    # plt.savefig("../../../fig/dbf/fscan_echo.png", dpi=300)

    image = fscan_sim.focus.wk_focus(echo, fscan_sim.R0)
    image_show = np.abs(image)/np.max(np.max(np.abs(image)))
    image_show = 20*np.log10(image_show)

    # image_fft = cp.abs(cp.fft.fftshift(cp.fft.fft2(cp.array(image)), axes=1)).get()
    # image_fft = image_fft/np.max(np.max(image_fft))
    # image_fft = 20*np.log10(image_fft)  
    # plt.figure()
    # plt.imshow(((image_fft)), aspect='auto', cmap='jet', vmin=-40, vmax=0, 
    #            extent=[-fscan_sim.Fs/2e6, fscan_sim.Fs/2e6, -fscan_sim.PRF/2, fscan_sim.PRF/2])
    # plt.colorbar()

    # plt.xlabel("Range Frequency(MHz)")
    # plt.ylabel("Azimuth Frequency(Hz)")
    # plt.savefig("../../../fig/dbf/fscan_echo_fft.png", dpi=300)


    plt.figure()
    plt.imshow(image_show, aspect="auto", cmap='jet', vmin=-40, vmax=0)
    plt.colorbar()
    plt.savefig("../../../fig/dbf/fscan_image.png", dpi=300)

    # Find the position of the maximum point in the image
    max_index = np.unravel_index(np.argmax(np.abs(image)), image.shape)
    fscan_sim.log.info("Position of the maximum point in the image:", max_index)
    target = image[max_index[0]-100:max_index[0]+100, max_index[0]-100:max_index[0]+100]
    target_up = fscan_sim.upsample(cp.array(target), (16, 16))
    image_show = np.abs(target)/np.max(np.max(np.abs(target)))
    image_show = 20*np.log10(image_show)
    
    plt.figure()
    plt.imshow(image_show, aspect="auto", cmap='jet', vmin=-40, vmax=0)
    plt.colorbar()
    plt.savefig("../../../fig/dbf/fscan_upimage.png", dpi=300)

    target = target_up

    fscan_range_res, fscan_index = fscan_sim.get_range_IRW(np.abs(target))
    fscan_rtarget = np.abs(target[fscan_index, :])
    fscan_rtarget = fscan_rtarget/np.max(fscan_rtarget)
    x = range(np.shape(fscan_rtarget)[0])
    plt.figure()
    plt.plot(x, 20*np.log10(fscan_rtarget))
    plt.savefig("../../../fig/dbf/fscan_range.png", dpi=300)

    fscan_azimuth_res, fscan_index = fscan_sim.get_azimuth_IRW(np.abs(target))
    fscan_atarget = np.abs(target[:, fscan_index])
    fscan_atarget = fscan_atarget/np.max(fscan_atarget)
    x = range(np.shape(fscan_atarget)[0])
    plt.figure()
    plt.plot(x, 20*np.log10(fscan_atarget))
    plt.savefig("../../../fig/dbf/fscan_azimuth.png", dpi=300)
    
    fscan_sim.log.info("fscan range irw: {}".format(fscan_range_res))
    fscan_sim.log.info("theoretical range irw: {}".format(fscan_sim.c/(2*Be)))
    fscan_sim.log.info("fscan azimuth irw: {}".format(fscan_azimuth_res))
    fscan_sim.log.info("theoretical azimuth irw: {}".format(fscan_sim.La/2))
    fscan_sim.log.info("fscan range pslr: {}".format(fscan_sim.get_pslr(fscan_rtarget)))
    fscan_sim.log.info("fscan azimuth pslr: {}".format(fscan_sim.get_pslr(fscan_atarget)))


def fscan_estimate():
    fscan_sim = Fscan()
    fscan_sim.set_B(2e9)
    fscan_sim.set_f0(35e9)

    fscan_sim.set_d(0.013)
    fscan_sim.set_N(13)
    fscan_sim.log.info("fscan beam width: {}".format(np.rad2deg(fscan_sim.fscan_beam_width)))
    fscan_sim.log.info("vr {}".format(fscan_sim.Vr))
    fscan_sim.log.info("max N {}".format(fscan_sim.dr*4*fscan_sim.f0/fscan_sim.c))
    fscan_sim.log.info("subaperture length {}".format(fscan_sim.d))
    # fscan_sim.dr = 0.14
    prf = np.linspace(500, 4e3, 3500)
    fscan_sim.zebra_diagram(prf, 1e-6)
    doa = np.linspace(fscan_sim.scan_left, fscan_sim.scan_right, 3000)
    fscan_sim.set_ttd(fscan_sim.get_ttd_rasr(doa))

    t_peak,_,_,_ = fscan_sim.calculate_doaTx(np.array([fscan_sim.beta]))
    fscan_sim.log.info("center t_peak: {}".format(t_peak))
  
    peak, left, right, bw = fscan_sim.calculate_doaTx(doa)
    fscan_sim.log.info("beam scan from {} us to {} us".format(np.min(left)*1e6, np.max(right)*1e6))

    pu = 250
    nesz_fscan = fscan_sim.nesz(doa, pu)
    plt.figure()
    plt.plot(np.rad2deg(doa), nesz_fscan, label="F-SCAN")
    plt.legend()
    plt.xlabel("视角/°", fontproperties=my_font)
    plt.ylabel("NESZ/dB")
    plt.grid()
    plt.tight_layout()
    plt.savefig("../../../fig/dbf/nesz.png", dpi=300)

    rasr_fscan = fscan_sim.rasr(doa)
    plt.figure()
    plt.plot(np.rad2deg(doa), rasr_fscan, label="F-SCAN")
    plt.legend()
    plt.xlabel("视角/°", fontproperties=my_font)
    plt.ylabel("RASR/dB")
    plt.grid()
    plt.tight_layout()
    plt.savefig("../../../fig/dbf/rasr.png", dpi=300)

    res_fscan = fscan_sim.resolution(doa)
    plt.figure()
    plt.plot(np.rad2deg(doa), res_fscan, label="F-SCAN")
    plt.legend()
    plt.xlabel("视角/°", fontproperties=my_font)
    plt.ylabel("距离向分辨率/m", fontproperties=my_font)
    plt.grid()
    plt.tight_layout()
    plt.savefig("../../../fig/dbf/resolution.png", dpi=300)

    prf = np.linspace(500, 2000, 500)
    aasr = fscan_sim.aasr(prf, 1)
    plt.figure()
    plt.plot(prf, aasr)
    plt.xlabel("PRF/Hz")
    plt.ylabel("AASR/dB")
    plt.grid()
    plt.tight_layout()
    plt.savefig("../../../fig/dbf/aasr.png", dpi=300)
    
    fscan_sim.swath_estimate(doa)
    fscan_sim.beam_pattern(doa)

def fscan_ant(fscan: Fscan):
    doa = np.linspace(fscan.scan_left , fscan.scan_right, 3000)
    N = np.arange(2, 50, 1)
    N_valid = []
    rasr = []
    nesz = []
    res_max = []
    res_min = []
    bw = []
    for n in N:
        fscan.set_N(n)
        fscan.set_ttd(fscan.get_ttd_rasr(doa))
        if fscan.ttd_judge(doa) == False:
            continue
        # doa_sin = np.sin(doa-fscan.beta)
        # the_m = 4*fscan.dr/(fscan.N*fscan.lambda_)
        # the_d = np.mean((the_m*fscan.c*fscan.B-fscan.c*(fscan.f0+fscan.B/2))/((np.max(doa_sin)-np.min(doa_sin))*(fscan.f0**2-fscan.B**2/4)))
        # fscan.log.debug("the d {} N {}".format(the_d, fscan.N))
        N_valid.append(n)
        f = fscan.calculate_doaf(doa)
        f_interval = np.abs(1/(fscan.N*(fscan.ttd-fscan.d*np.sin(doa-fscan.beta)/fscan.c)))
        band_width = np.max(f+f_interval) - np.min(f-f_interval)
        bw.append(band_width)
        rasr.append(np.max(fscan.rasr(doa)))
        nesz.append(np.max(fscan.nesz(doa, 5e3)))
        res_max.append(np.max(fscan.resolution(doa)))
        res_min.append(np.min(fscan.resolution(doa)))
    rasr = np.array(rasr)
    nesz = np.array(nesz)
    res_max = np.array(res_max)
    res_min = np.array(res_min)
    N_valid = np.array(N_valid)
    bw = np.array(bw)

    return N_valid, rasr, nesz, res_max, bw


def fscan_ant_d_estimate():
    fscan = Fscan()
    fscan.set_B(2e9)
    fscan.set_f0(35e9)
    fscan.set_scanwidth(np.deg2rad(10))
    fscan.dr= 0.14
    d = [0.006, 0.008, 0.01]
    N_valid = []
    rasr = []
    nesz = []
    bw = []
    for di in d:
        fscan.set_d(di)
        res = fscan_ant(fscan)
        N_valid.append(res[0])
        rasr.append(res[1])
        nesz.append(res[2])
        bw.append(res[4])

    plt.figure()
    for i in range(len(d)):
        plt.plot(N_valid[i], rasr[i], label="d = {}m".format(d[i]), marker='o')
    plt.xlabel("N")
    plt.ylabel("RASR/dB")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("../../../fig/dbf/fscan_ant_drasr.png", dpi=300)
    plt.figure()
    for i in range(len(d)):
        plt.plot(N_valid[i], nesz[i], label="d = {}m".format(d[i]), marker='o')
    plt.xlabel("N")
    plt.ylabel("NESZ/dB")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("../../../fig/dbf/fscan_ant_dnesz.png", dpi=300)
    plt.figure()
    for i in range(len(d)):
        plt.plot(N_valid[i], bw[i], label="d = {}m".format(d[i]), marker='o')
    plt.xlabel("N")
    plt.ylabel("带宽/MHz", fontproperties=my_font)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("../../../fig/dbf/fscan_ant_dbw.png", dpi=300)


def fscan_ant_f_estimate():
    fscan = Fscan()
    fscan.set_B(4e9)
    fscan.set_d(0.01)
    fscan.set_f0(9.8e9)
    fscan.set_scanwidth(np.deg2rad(10))
    fscan.dr = 0.2
    N_valid1, rasr1, nesz1, _ , bw = fscan_ant(fscan)
    fscan.set_f0(32e9)
    N_valid2, rasr2, nesz2, _, bw = fscan_ant(fscan)

    plt.figure()
    plt.plot(N_valid1, rasr1, label="f = 9.8GHz", marker='^')
    plt.plot(N_valid2, rasr2, label="f = 32GHz", marker='o')
    plt.xlabel("N")
    plt.ylabel("RASR/dB")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("../../../fig/dbf/fscan_ant_rasr.png", dpi=300)
    plt.figure()
    plt.plot(N_valid1, nesz1, label="f = 9.8GHz", marker='^')
    plt.plot(N_valid2, nesz2, label="f = 32GHz", marker='o')
    plt.xlabel("N")
    plt.ylabel("NESZ/dB")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("../../../fig/dbf/fscan_ant_nesz.png", dpi=300)



def fscan_carrier_estimate():
    fscan = Fscan()
    fscan.set_scanwidth(np.deg2rad(4))
    fscan.set_d(0.03)
    fscan.set_B(2e9)
    fscan.set_N(10)
    fscan.dr = 0.2
    doa = np.linspace(fscan.scan_left, fscan.scan_right, 3000)
    fscan.set_ttd(fscan.get_ttd_bandwidth(doa))

    carrier = np.arange(9e9, 40e9, 5e8)

    scan = []
    carrier_valid = []
    ttd = []
    for c in carrier:
        fscan.set_f0(c)
        scan_left =  np.arcsin((fscan.ttd*fscan.c-np.round(c*fscan.ttd)*fscan.c/(c+fscan.B/2))/fscan.d)
        scan_right = np.arcsin((fscan.ttd*fscan.c-np.round(c*fscan.ttd)*fscan.c/(c-fscan.B/2))/fscan.d)
        scan_width = np.abs(scan_left-scan_right)
        scan_left = fscan.beta - scan_width/2
        scan_right = fscan.beta + scan_width/2
        doa = np.linspace(fscan.scan_left, fscan.scan_right, 1000)
        fscan.set_ttd(fscan.get_ttd_bandwidth(doa))
        if fscan.ttd_judge(doa) == False:
            continue
        carrier_valid.append(c)
        m = np.min(fscan.calculate_m(doa))
        scan_left =  np.arcsin((fscan.ttd*fscan.c-m*fscan.c/(c+fscan.B/2))/fscan.d)
        scan_right = np.arcsin((fscan.ttd*fscan.c-m*fscan.c/(c-fscan.B/2))/fscan.d)
        scan.append(np.abs(scan_left-scan_right))
        ttd.append(fscan.ttd)

    carrier = np.array(carrier_valid)
    scan = np.array(scan)
    ttd = np.array(ttd)
    ttd_use = np.mean(ttd)
    scan_my_left =  np.arcsin((fscan.ttd*fscan.c-(carrier*ttd_use)*fscan.c/(carrier+fscan.B/2))/fscan.d)
    scan_my_right = np.arcsin((fscan.ttd*fscan.c-(carrier*ttd_use)*fscan.c/(carrier-fscan.B/2))/fscan.d)
    scan_my = np.abs(scan_my_left-scan_my_right)

    diff_scan_my = -fscan.B**2 * np.abs(fscan.ttd)*fscan.c/(fscan.d*carrier**3)
    diff_scan_left = (fscan.B*fscan.ttd*fscan.c/(2*fscan.d*(carrier+fscan.B/2)**2*np.sqrt(1-(fscan.ttd*fscan.c-(carrier*fscan.ttd*fscan.c)/(carrier+fscan.B/2))**2/fscan.d**2)))
    diff_scan_right = (fscan.B*fscan.ttd*fscan.c/(2*fscan.d*(carrier-fscan.B/2)**2*np.sqrt(1-(fscan.ttd*fscan.c-(carrier*fscan.ttd*fscan.c)/(carrier-fscan.B/2))**2/fscan.d**2)))
    diff_scan_my1 = -np.sign(fscan.ttd)*(-diff_scan_left+diff_scan_right)
    diff_scan = np.gradient(scan, carrier)

    plt.figure()
    plt.plot(carrier/1e9, np.rad2deg(scan), marker='o', label = "sim")
    plt.plot(carrier/1e9, np.rad2deg(scan_my), marker='^', label = "theory")
    plt.legend()
    plt.xlabel("载波频率/GHz", fontproperties=my_font)
    plt.ylabel("视角 width/°", fontproperties=my_font)
    plt.grid()
    plt.tight_layout()
    plt.savefig("../../../fig/dbf/fscan_carrier_scan.png", dpi=300)

    plt.figure()
    plt.plot(carrier/1e9, diff_scan, marker='o', label="simulation")
    plt.plot(carrier/1e9, diff_scan_my1, marker='^', label="theory")
    plt.xlabel("载波频率/GHz", fontproperties=my_font)
    plt.ylabel("视角 width/°", fontproperties=my_font)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("../../../fig/dbf/fscan_carrier_scan_diff.png", dpi=300)



def unsuitable():
    fscan = Fscan()
    fscan.set_B(2e9)
    fscan.set_d(0.01)
    fscan.set_groundwidth(50e3)
    doa = np.linspace(fscan.scan_left, fscan.scan_right, 3000)
    fscan.set_ttd(fscan.d*np.sin(fscan.beta)/fscan.c)
    fscan.beam_pattern(doa)
    fscan.log.info("ttd: {}".format(fscan.ttd))

if __name__ == '__main__':
    fscan_estimate()
    # fscan_carrier_estimate()
    # unsuitable()
    # fscan_ant_d_estimate()
    # fscan_ant_f_estimate()
    # fscan_simulation()
    # dbf_simulation()
