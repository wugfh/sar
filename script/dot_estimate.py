from os import path

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import cv2
import h5py
plt.rc("font", family="Times New Roman")
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 14

class DotEstimator:
    def __init__(self, point_n, c, Vr, PRF, Fs, path, H = 0, Rc = 0):
        self.points_n = point_n
        self.c = c
        self.Vr = Vr
        self.PRF = PRF
        self.Fs = Fs
        self.path = path
        self.Rc = Rc
        self.H = H


    def upsample(self, data, N):
        Na, Nr = np.shape(data)
        data_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(data)))
        tmp = np.zeros((N[0]*Na, N[1]*Nr), dtype=complex)
        tmp[N[0]*Na//2-Na//2:N[0]*Na//2+Na//2, N[1]*Nr//2-Nr//2:N[1]*Nr//2+Nr//2] = data_fft
        data_up = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(tmp)))
        return data_up

    def get_range_IRW(self, ehco, uprate):
        target =  np.max(np.abs(ehco), axis=0)
        max_value = np.max(target)
        half_max = max_value/np.sqrt(2)
        valid = np.abs(target) > half_max
        irw = np.sum(valid)
        irw = irw*self.c/(2*uprate*self.Fs)
        return irw
    
    def get_azimuth_IRW(self, ehco, uprate):
        target =  np.max(np.abs(ehco), axis=1)
        max_value = np.max(target)
        half_max = max_value/np.sqrt(2)
        valid = np.abs(target) > half_max
        irw = np.sum(valid)
        irw = irw*self.Vr/(self.PRF*uprate)
        return irw
    
    def get_pslr(self, target):
        target_np = (target)
        peaks, _ = signal.find_peaks(target_np)

        mainlobe_peak_index = peaks[np.argmax(target_np[peaks])]
        mainlobe_peak_value = target_np[mainlobe_peak_index]

        sidelobe_peaks = np.delete(peaks, np.argmax(target_np[peaks]))
        sidelobe_peak_value = np.max(target_np[sidelobe_peaks])

        pslr = 20 * np.log10(sidelobe_peak_value / mainlobe_peak_value)
        
        return pslr
    
    def get_islr(self, target):
        target_np = np.abs(target)**2
        total_power = np.sum(target_np)

        peaks, _ = signal.find_peaks(target_np)
        mainlobe_peak_index = peaks[np.argmax(target_np[peaks])]

        low_peaks,_ = signal.find_peaks(-target_np)
        left_peak = low_peaks[low_peaks < mainlobe_peak_index].max()
        right_peak = low_peaks[low_peaks > mainlobe_peak_index].min()
        mainlobe_power = np.sum(target_np[left_peak:right_peak])
        sidelobe_power = total_power - mainlobe_power
        islr = 10 * np.log10(sidelobe_power / mainlobe_power)
        
        return islr
    
    def dot_estimate(self, image, area, uprate):
        plt.figure(figsize=(4*self.points_n, 8))
        # Convert numerical values to corresponding letters
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        numerical_values = np.arange(len(alphabet))
        letter_mapping = dict(zip(numerical_values, alphabet))
        image_copy = image.copy()
        cnt = 0
        range_res = []
        range_pslr = []
        range_islr = []
        azimuth_res = []
        azimuth_pslr = []
        azimuth_islr = []
        look_angle = []
        point_pos = []
        while cnt < self.points_n:
           
            max_index = np.unravel_index(np.argmax(np.abs(image_copy)), image_copy.shape)
            if max_index[0] < area[0]//2 or max_index[0] > image_copy.shape[0]-area[0]//2 or max_index[1] < area[1]//2 or max_index[1] > image_copy.shape[1]-area[1]//2:
                print("The maximum point is out of the area:\n shape {}  maxindex:{}   area:{}.".format(image_copy.shape, max_index, area))
                up = np.maximum(0, max_index[0] - area[0]//2)
                down = np.minimum(image_copy.shape[0], max_index[0] + area[0]//2)
                left = np.maximum(0, max_index[1] - area[1]//2)
                right = np.minimum(image_copy.shape[1], max_index[1] + area[1]//2)
                image_copy[up:down, left:right] = 0
                continue
            # print("Position of the maximum point in the image:", max_index)
            target = image_copy[max_index[0]-area[0]//2:max_index[0]+area[0]//2, max_index[1]-area[1]//2:max_index[1]+area[1]//2]
            target_up = self.upsample(np.array(target), (uprate, uprate))

            look_angle.append(self.get_look_angle(max_index[1], image_copy.shape[1]))

            target = target_up
            x = np.array([-area[1]/2, area[1]/2])
            dr = x*self.c/(2*self.Fs)
            y =  np.array([-area[0]/2, area[0]/2])
            da = y*self.Vr/(self.PRF)


            fscan_range_res = self.get_range_IRW(np.abs(target), uprate)
            range_res.append(fscan_range_res)
            fscan_rtarget = np.max(np.abs(target), axis=0)
            fscan_rtarget = fscan_rtarget/np.max(fscan_rtarget)
            x_dr = np.linspace(dr[0], dr[1], len(fscan_rtarget))

            plt.subplot(3, self.points_n, self.points_n + cnt+1)
            # plt.figure()
            plt.plot(x_dr, 20*np.log10(fscan_rtarget))
            plt.grid()
        
            plt.ylim(-30, 0)
            plt.xlabel("range (m)", fontsize=18, fontweight='bold')
            plt.ylabel("amplitude (dB)", fontsize=18, fontweight='bold')
            ax = plt.gca()
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')
                label.set_fontsize(14)
            # plt.savefig(self.path+"range.png", dpi=300)
            # plt.title("({}{})".format(letter_mapping[(self.points_n + cnt+1)%26],1))

            fscan_azimuth_res = self.get_azimuth_IRW(np.abs(target), uprate)
            azimuth_res.append(fscan_azimuth_res)
            fscan_atarget = np.max(np.abs(target), axis=1)
            fscan_atarget = fscan_atarget/np.max(fscan_atarget)
            x_da = np.linspace(da[0], da[1], len(fscan_atarget))

            plt.subplot(3, self.points_n, 2*self.points_n + cnt+1)
            # plt.figure()
            plt.plot(x_da, 20*np.log10(fscan_atarget))
            plt.grid()
            plt.ylim(-30, 0)
            plt.xlabel("azimuth (m)", fontsize=18, fontweight='bold')
            plt.ylabel("amplitude (dB)", fontsize=18, fontweight='bold')
            ax = plt.gca()
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')
                label.set_fontsize(14)
            # plt.savefig(self.path+"azimuth.png", dpi=300)
            # plt.title("({}{})".format(letter_mapping[(self.points_n + cnt+1)%26],2))

            image_show = np.abs(target_up)/np.max(np.abs(target_up))
            image_show = 20*np.log10(image_show)  
            image_show[image_show < -60] = -60          
            plt.subplot(3, self.points_n, cnt+1)
            # plt.figure()
            # plt.imshow(np.abs(tmp), aspect="auto", cmap='jet', extent=[dr[0], dr[1], da[0], da[1]])
            plt.imshow(image_show, aspect="auto", cmap='jet', extent=[dr[0], dr[1], da[0], da[1]], vmax = 0, vmin = -60)
            plt.ylabel("azimuth(m)", fontsize=18, fontweight='bold')
            plt.xlabel("range(m)", fontsize=18, fontweight='bold')
            ax = plt.gca()
            colorbar = plt.colorbar()
            colorbar.ax.set_title("dB", fontsize=18, fontweight='bold')
            # range_vals = np.linspace(dr[0], dr[1], image_show.shape[1])  # 列数
            # az_vals   = np.linspace(da[0], da[1], image_show.shape[0])  # 行数
            # R, A = np.meshgrid(range_vals, az_vals)

            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')

            # surf = ax.plot_surface(R, A, image_show,
            #                     cmap='jet',  
            #                     antialiased=True,
            #                     rstride=1, cstride=1,      
            #                     alpha=None,  
            #                     )

            # ax.view_init(elev=30, azim=-37.5)

            # # 轴标签与标题
            # ax.set_xlabel('range (m)', fontsize=18, fontweight='bold')
            # ax.set_ylabel('azimuth (m)', fontsize=18, fontweight='bold')
            # ax.set_zlabel('amplitude (dB)', fontsize=18, fontweight='bold')
            for label in ax.get_xticklabels() + ax.get_yticklabels() + colorbar.ax.get_yticklabels():
                label.set_fontweight('bold')
                label.set_fontsize(14)
            # ax.set_zlabel('dB')

            # ax.set_zlim(-60, 0)

            # cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=12)
            # cbar.ax.set_title('dB')

            # ax.xaxis.pane.fill = False
            # ax.yaxis.pane.fill = False
            # ax.zaxis.pane.fill = False
            # ax.xaxis.pane.set_edgecolor('w')
            # ax.yaxis.pane.set_edgecolor('w')
            # ax.zaxis.pane.set_edgecolor('w')
            # plt.subplots_adjust(left=0.05, right=0.95, top=1, bottom=0)

            # plt.savefig(self.path+"dot.png", dpi=300)
            # plt.title("({}{})".format(letter_mapping[(cnt+1)%26],0))


            # print("range irw: ", fscan_range_res)
            # print("azimuth irw: ", fscan_azimuth_res)
            # print("range pslr: ", self.get_pslr(fscan_rtarget))
            # print("azimuth pslr: ", self.get_pslr(fscan_atarget))
            range_pslr.append(self.get_pslr(fscan_rtarget))
            # azimuth_pslr.append(self.get_pslr(fscan_atarget))
            range_islr.append(self.get_islr(fscan_rtarget))
            # azimuth_islr.append(self.get_islr(fscan_atarget))
            # print("range islr: ", self.get_islr(fscan_rtarget))
            # print("azimuth islr: ", self.get_islr(fscan_atarget))
            if range_pslr[-1] < -8:
                point_pos.append(max_index)

            image_copy[max_index[0]-area[0]//2:max_index[0]+area[0]//2, max_index[1]-area[1]//2:max_index[1]+area[1]//2] = 0
            cnt = cnt+1
        
        
        plt.tight_layout()
        # adjust spacing between subplots
        # plt.subplots_adjust(hspace=0.3, wspace=0.2)

        plt.savefig(self.path+"dot_estimate.png", dpi=300)
        range_res = np.array(range_res)
        range_pslr = np.array(range_pslr)
        range_islr = np.array(range_islr)
        # azimuth_res = np.array(azimuth_res)
        # azimuth_pslr = np.array(azimuth_pslr)
        # azimuth_islr = np.array(azimuth_islr)
        look_angle = np.array(look_angle)
        look_angle = look_angle[range_pslr < -8]
        range_res = range_res[range_pslr < -8]
        range_islr = range_islr[range_pslr < -8]
        range_pslr = range_pslr[range_pslr < -8]

        # azimuth_res = azimuth_res[azimuth_pslr < -8]
        # azimuth_islr = azimuth_islr[azimuth_pslr < -8]
        # azimuth_pslr = azimuth_pslr[azimuth_pslr < -8]


        print("range resolution: {} m".format(range_res.mean()))
        print("range resolution std: {} m".format(range_res.std()))
        print("range pslr: {} dB".format(range_pslr.mean()))
        print("range pslr std: {} dB".format(range_pslr.std()))
        print("range islr: {} dB".format(range_islr.mean()))
        print("range islr std: {} dB".format(range_islr.std()))

        # print("azimuth resolution: {} m".format(azimuth_res.mean()))
        # print("azimuth resolution std: {} m".format(azimuth_res.std()))
        # print("azimuth pslr: {} dB".format(azimuth_pslr.mean()))
        # print("azimuth pslr std: {} dB".format(azimuth_pslr.std()))
        # print("azimuth islr: {} dB".format(azimuth_islr.mean()))
        # print("azimuth islr std: {} dB".format(azimuth_islr.std()))

        print("check the number of valid points: {}".format(len(range_res)))
        return range_res, look_angle, point_pos

    def pslr_estimate(self, image, area, uprate):
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        numerical_values = np.arange(len(alphabet))
        letter_mapping = dict(zip(numerical_values, alphabet))
        image_copy = image.copy()
           
        max_index = np.unravel_index(np.argmax(np.abs(image_copy)), image_copy.shape)
        if max_index[0] < area[0]//2 or max_index[0] > image_copy.shape[0]-area[0]//2 or max_index[1] < area[1]//2 or max_index[1] > image_copy.shape[1]-area[1]//2:
            print("The maximum point is out of the area:\n shape {}  maxindex:{}   area:{}.".format(image_copy.shape, max_index, area))

        target = image_copy[max_index[0]-area[0]//2:max_index[0]+area[0]//2, max_index[1]-area[1]//2:max_index[1]+area[1]//2]
        target_up = self.upsample(np.array(target), (uprate, uprate))


        target = target_up
        fscan_atarget = np.max(np.abs(target), axis=1)
        fscan_atarget = fscan_atarget/np.max(fscan_atarget)

        pslr = self.get_pslr(fscan_atarget)
        return pslr

    def time_frequency_estimate(self, data):
        # reverse the input data
        sig = data.squeeze()

        nperseg = min(256, sig.size)
        noverlap = nperseg // 2

        f, t, Zxx = signal.stft(
            sig,
            fs=self.Fs,
            window='hann',
            nperseg=nperseg,
            noverlap=noverlap,
            return_onesided=False
        )

        f = np.fft.fftshift(f)
        f = np.flip(f)
        Zxx = np.fft.fftshift(Zxx, axes=0)

        spec_db = 20 * np.log10(np.abs(Zxx) + 1e-12)
        plt.figure(figsize=(8, 4))
        plt.pcolormesh(t*1e6, f/1e9, spec_db, shading='gouraud', cmap='jet')
        plt.xlabel('time (us)', fontsize=18, fontweight='bold')
        plt.ylabel('frequency (GHz)', fontsize=18, fontweight='bold')
        cb = plt.colorbar(label='amplitude (dB)')
        cb.set_label('amplitude (dB)', fontsize=18, fontweight='bold')


        ax = plt.gca()
        for label in ax.get_xticklabels() + ax.get_yticklabels() + cb.ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_fontsize(14)
        plt.tight_layout()
        plt.savefig(self.path + "stft.pdf", dpi=1000)

    def get_look_angle(self, pos, Nr):
        Rp = pos*self.c/(2*self.Fs)+self.Rc-Nr*self.c/(4*self.Fs)
        angle = np.arccos(self.H/Rp)
        return angle


if __name__ == "__main__":
    single_vr = 63.5
    fscan_vr = 70.2
    c = 299792458
    PRF = 1600
    fs = 2.5e9
    H = 2922
    phi = np.rad2deg(61.83)
    Rc = 6371
    recovery_estimator = DotEstimator(point_n=50, c=c, Vr=fscan_vr, PRF=PRF, Fs=fs, path="../fig/afscan/recovery_", H=H, Rc=Rc)
    fscan_estimator = DotEstimator(point_n=50, c=c, Vr=fscan_vr, PRF=PRF, Fs=fs, path="../fig/afscan/fscan_", H=H, Rc=Rc)


    

    fscan_path = "../fig/afscan/par_focus_super_100_0.mat"
    recovery_path = "../fig/afscan/par_focus_super_100_200.mat"
    import scipy.io as sio
    fscan = sio.loadmat(fscan_path)['focus']
    recovery = sio.loadmat(recovery_path)['focus']
    # with h5py.File(fscan_path, "r") as data:
    #     fscan = data['sig']
    #     fscan = np.array(fscan)
    # with h5py.File(recovery_path, "r") as data:
    #     recovery = data['sig']
    #     recovery = np.array(recovery)


    # fscan_estimator.time_frequency_estimate(fscan[12500, :])
    # single_estimator.time_frequency_estimate(single[13000, :])

 

    area = (int(1/(fscan_vr/PRF)), int(3/(c/(2*fs))))
    print("area: ", area)

    res = np.array([])
    look_angle = np.array([])
    Nr = fscan.shape[1]
    fig, ax = plt.subplots()
    block_array = np.array([0, 1])
    for i in range(block_array.shape[0]-1):
        start = int(block_array[i]*fscan.shape[1])
        end = int(block_array[i+1]*fscan.shape[1])
        fscan_estimator.Rc = Rc - Nr/2*c/(2*fs) + (start+end)/2*c/(2*fs)
        print("Rc: ", fscan_estimator.Rc)
        r, angle, pos = fscan_estimator.dot_estimate(fscan[:, start:end], area, 16)
        res = np.concatenate((res, r))
        look_angle = np.concatenate((look_angle, angle))
        print("\r\n")
        # single_estimator.dot_estimate(single[:, start:end], area, 16)

        print("point cnt", len(pos))
        image = fscan[:, start:end]
        select_pos = pos[1]
        target = image[select_pos[0]-area[0]//2:select_pos[0]+area[0]//2, select_pos[1]-area[1]//2:select_pos[1]+area[1]//2]
        target_up = fscan_estimator.upsample(np.array(target), (16, 16))

        target = target_up
        x = np.array([-area[1]/2, area[1]/2])
        dr = x*fscan_estimator.c/(2*fscan_estimator.Fs)
        y =  np.array([-area[0]/2, area[0]/2])
        da = y*fscan_estimator.Vr/(fscan_estimator.PRF)

        label_text = "fscan"
        color = "#0072B2"
        style = ":"

        fscan_rtarget = np.max(np.abs(target), axis=0)
        fscan_rtarget = fscan_rtarget/np.max(fscan_rtarget)
        x_dr = np.linspace(dr[0], dr[1], len(fscan_rtarget))
        ax.plot(x_dr, 20*np.log10(fscan_rtarget), label=label_text, linewidth=2, color=color, linestyle=style)

        
        label_text = "recovery"
        color = "#CC3311"
        style = None

        start = int(block_array[i]*recovery.shape[1])
        end = int(block_array[i+1]*recovery.shape[1])
        recovery_estimator.Rc = Rc - Nr/2*c/(2*fs) + (start+end)/2*c/(2*fs)
        print("Rc: ", recovery_estimator.Rc)
        r, angle, pos = recovery_estimator.dot_estimate(recovery[:, start:end], area, 16)
        res = np.concatenate((res, r))
        look_angle = np.concatenate((look_angle, angle))
        print("\r\n")
        # single_estimator.dot_estimate(single[:, start:end], area, 16)

        print("point cnt", len(pos))
        image = recovery[:, start:end]
        select_pos = pos[1]
        target = image[select_pos[0]-area[0]//2:select_pos[0]+area[0]//2, select_pos[1]-area[1]//2:select_pos[1]+area[1]//2]
        target_up = recovery_estimator.upsample(np.array(target), (16, 16))

        target = target_up
        x = np.array([-area[1]/2, area[1]/2])
        dr = x*recovery_estimator.c/(2*recovery_estimator.Fs)
        y =  np.array([-area[0]/2, area[0]/2])
        da = y*recovery_estimator.Vr/(recovery_estimator.PRF)

        fscan_rtarget = np.max(np.abs(target), axis=0)
        fscan_rtarget = fscan_rtarget/np.max(fscan_rtarget)
        x_dr = np.linspace(dr[0], dr[1], len(fscan_rtarget))
        ax.plot(x_dr, 20*np.log10(fscan_rtarget), label=label_text, linewidth=2, color=color, linestyle=style)

    ax.grid(True, which='major', color='0.75', ls=':', lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.tick_params(which='major', length=3.5, width=0.8)
    ax.tick_params(which='minor', length=2.0, width=0.6)
    ax.margins(x=0.01, y=0.05)
    ax.set_xlabel("range (m)", fontsize=18, fontweight='bold')
    ax.set_ylabel("amplitude (dB)", fontsize=18, fontweight='bold')
    ax.legend(loc='best', ncol=2, frameon=True, framealpha=0.9,
            edgecolor='0.7', borderpad=0.4, handlelength=2.2, columnspacing=1.0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(14)
        label.set_fontweight('bold')

    fig.tight_layout()
    fig.savefig('recovery_sidelode.pdf', dpi=300, bbox_inches='tight')


    # import scipy.io as sio
    # sio.savemat("../fig/afscan/fscan_dot_estimate.mat", {"range_res": res, "look_angle": look_angle})

    # single_estimator.dot_estimate(single[9300:9600, 4900:5100], (int(1/(single_vr/PRF)), int(3/(c/(2*fs)))), 16)
    fscan_bench = np.abs(fscan[200:300, 2259:2330]).mean()
    recovery_bench = np.abs(recovery[200:300, 2259:2330]).mean()

    # print("shape of fscan: ", fscan.shape)
    # print("shape of recovery: ", recovery.shape)
    fscan_power_mean = np.percentile(np.abs(fscan), 75, axis=0)
    recovery_power_mean = np.percentile(np.abs(recovery), 75, axis=0)
    print("shape of fscan power mean: ", fscan_power_mean.shape)
    fscan_snr = 20*np.log10(fscan_power_mean/fscan_bench)
    recovery_snr = 20*np.log10(recovery_power_mean/recovery_bench)



    H = 2922
    phi = np.rad2deg(61.83)
    Rc = 6371
    range_swath = (np.arange(fscan.shape[1])-fscan.shape[1]//2) * c / (2 * fs) + Rc
    doa = np.arccos(H/range_swath)
    ground_swath = H*np.tan(doa)
    ground_swath = ground_swath - ground_swath[0]

    ksize = 50/(ground_swath[1] - ground_swath[0])
    fscan_snr = np.convolve(fscan_snr, np.ones(int(ksize))/int(ksize), mode='same')
    recovery_snr = np.convolve(recovery_snr, np.ones(int(ksize))/int(ksize), mode='same')

    th = np.arange(3, 12, 1)
    fscan_swath_list = []
    recovery_swath_list = []
    
    for th in range(3, 12):
        fscan_thresh = fscan_snr > th
        recovery_thresh = recovery_snr > th
        fscan_left = 0
        fscan_right = 0
        recovery_left = 0
        recovery_right = 0
        for i in range(len(fscan_thresh)-1):
            if fscan_left == 0 and fscan_thresh[i] == True:
                fscan_left = i
            if recovery_left == 0 and recovery_thresh[i] == True:
                recovery_left = i
        for i in range(len(fscan_thresh)-1, 0, -1):
            if fscan_right == 0 and fscan_thresh[i] == True:
                fscan_right = i
            if recovery_right == 0 and recovery_thresh[i] == True:
                recovery_right = i
        fscan_swath = ground_swath[fscan_right] - ground_swath[fscan_left]
        recovery_swath = ground_swath[recovery_right] - ground_swath[recovery_left]
        fscan_swath_list.append(fscan_swath)
        recovery_swath_list.append(recovery_swath)
    print("max SNR: fscan: {} dB, recovery: {} dB".format(np.max(fscan_snr), np.max(recovery_snr)))

    plt.figure()
    plt.plot(ground_swath, fscan_snr, label="FSAR", linewidth=2)
    plt.plot(ground_swath, recovery_snr, label="Recovery", linewidth=2)
    plt.axhline(y=4, color='k', linestyle='--', label="4 dB SNR threshold")
    plt.xlabel("Ground (m)", fontsize=18, fontweight='bold')
    plt.ylabel("SNR (dB)", fontsize=18, fontweight='bold')
    plt.legend(fontsize=14)
    plt.grid()
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(14)
    plt.savefig("../fig/afscan/snr_recovery.pdf", dpi=300)


