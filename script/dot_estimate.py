import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import cv2
import h5py
class DotEstimator:
    def __init__(self, point_n, c, Vr, PRF, Fs, path):
        self.points_n = point_n
        self.c = c
        self.Vr = Vr
        self.PRF = PRF
        self.Fs = Fs
        self.path = path
        
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
        while cnt < self.points_n:
           
            max_index = np.unravel_index(np.argmax(np.abs(image_copy)), image_copy.shape)
            if max_index[0] < area[0]//2 or max_index[0] > image_copy.shape[0]-area[0]//2 or max_index[1] < area[1]//2 or max_index[1] > image_copy.shape[1]-area[1]//2:
                cnt = cnt+1
                print("The maximum point is out of the area:\n shape {}  maxindex:{}   area:{}.".format(image_copy.shape, max_index, area))
                continue

            print("Position of the maximum point in the image:", max_index)
            target = image_copy[max_index[0]-area[0]//2:max_index[0]+area[0]//2, max_index[1]-area[1]//2:max_index[1]+area[1]//2]
            target_up = self.upsample(np.array(target), (uprate, uprate))



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

            # plt.subplot(3, self.points_n, self.points_n + cnt+1)
            plt.figure()
            plt.plot(x_dr, 20*np.log10(fscan_rtarget))
            plt.grid()
            plt.ylim(-30, 0)
            plt.xlabel("range(m)")
            plt.ylabel("amplitude(dB)")
            plt.savefig(self.path+"range.png", dpi=300)
            # plt.title("({}{})".format(letter_mapping[(self.points_n + cnt+1)%26],1))

            fscan_azimuth_res = self.get_azimuth_IRW(np.abs(target), uprate)
            fscan_atarget = np.max(np.abs(target), axis=1)
            fscan_atarget = fscan_atarget/np.max(fscan_atarget)
            x_da = np.linspace(da[0], da[1], len(fscan_atarget))

            # plt.subplot(3, self.points_n, 2*self.points_n + cnt+1)
            plt.figure()
            plt.plot(x_da, 20*np.log10(fscan_atarget))
            plt.grid()
            plt.ylim(-30, 0)
            plt.xlabel("azimuth(m)")
            plt.ylabel("amplitude(dB)")
            plt.savefig(self.path+"azimuth.png", dpi=300)
            # plt.title("({}{})".format(letter_mapping[(self.points_n + cnt+1)%26],2))

            image_show = np.abs(target_up)/np.max(np.abs(target_up))
            image_show = 20*np.log10(image_show)  
            image_show[image_show < -60] = -60          
            # plt.subplot(3, self.points_n, cnt+1)
            plt.figure()
            # # plt.imshow(np.abs(tmp), aspect="auto", cmap='jet', extent=[dr[0], dr[1], da[0], da[1]])
            # plt.imshow(image_show, aspect="auto", cmap='jet', extent=[dr[0], dr[1], da[0], da[1]], vmax = 0, vmin = -60)
            # plt.ylabel("azimuth(m)")
            # plt.xlabel("range(m)")
            # colorbar = plt.colorbar()
            # colorbar.ax.set_title("dB")
            range_vals = np.linspace(dr[0], dr[1], image_show.shape[1])  # 列数
            az_vals   = np.linspace(da[0], da[1], image_show.shape[0])  # 行数
            R, A = np.meshgrid(range_vals, az_vals)

            # 设置全局字体（可选，接近 MATLAB 默认字体）
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']  # MATLAB 常用 Arial
            plt.rcParams['font.size'] = 10

            # 创建 3D 图
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # 绘制曲面（保留网格线）
            surf = ax.plot_surface(R, A, image_show,
                                cmap='jet',  
                                antialiased=True,
                                rstride=1, cstride=1,      
                                alpha=None,  
                                )

            # MATLAB 默认视角：方位角 -37.5°，仰角 30°
            ax.view_init(elev=30, azim=-37.5)

            # 轴标签与标题
            ax.set_xlabel('range (m)')
            ax.set_ylabel('azimuth (m)')
            ax.set_zlabel('amplitude (dB)')
            # ax.set_zlabel('dB')

            # 设置 Z 轴范围
            # ax.set_zlim(-60, 0)

            # # 颜色条（紧凑显示）
            # cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=12)
            # cbar.ax.set_title('dB')

            # 将坐标轴背景板设为透明（类似 MATLAB 白色背景无遮挡）
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('w')
            ax.yaxis.pane.set_edgecolor('w')
            ax.zaxis.pane.set_edgecolor('w')

            # 可选：调整坐标轴数据比例接近 MATLAB 的“tight”效果
            # MATLAB 默认不会强制等轴，但可以用 set_box_aspect 微调
            # ax.set_box_aspect((1, 0.8, 0.6))  # 可根据实际数据调整
            plt.subplots_adjust(left=0.05, right=0.95, top=1, bottom=0)

            plt.savefig(self.path+"dot.png", dpi=300)
            # plt.title("({}{})".format(letter_mapping[(cnt+1)%26],0))


            print("range irw: ", fscan_range_res)
            print("azimuth irw: ", fscan_azimuth_res)
            print("range pslr: ", self.get_pslr(fscan_rtarget))
            print("azimuth pslr: ", self.get_pslr(fscan_atarget))
            print("range islr: ", self.get_islr(fscan_rtarget))
            print("azimuth islr: ", self.get_islr(fscan_atarget))

            image_copy[max_index[0]-area[0]//2:max_index[0]+area[0]//2, max_index[1]-area[1]//2:max_index[1]+area[1]//2] = 0
            cnt = cnt+1
        
        plt.tight_layout()

        # plt.savefig(self.path+"dot_estimate.png", dpi=300)
        range_res = np.array(range_res)
        print("range resolution: {} m".format(range_res.mean()))
        return range_res.mean()

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
        sig = data[::-1].squeeze()

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
        plt.figure()
        plt.pcolormesh(t*1e6, f/1e9, spec_db, shading='gouraud', cmap='jet')
        plt.xlabel('time (us)')
        plt.ylabel('frequency (GHz)')
        plt.colorbar(label='amplitude (dB)')
        plt.tight_layout()
        plt.savefig(self.path + "stft.pdf", dpi=1000)

        api = "sk-d335b12ad5db414aa010b0651992183e"


if __name__ == "__main__":
    single_vr = 63.5
    fscan_vr = 70.2
    c = 299792458
    PRF = 1600
    fs = 2.5e9
    single_estimator = DotEstimator(point_n=1, c=c, Vr=single_vr, PRF=PRF, Fs=fs, path="../fig/afscan/single_")
    fscan_estimator = DotEstimator(point_n=1, c=c, Vr=fscan_vr, PRF=PRF, Fs=fs, path="../fig/afscan/fscan_")

    fscan_path = "../fig/afscan/example_10_part6_focus.mat"
    single_path = "../fig/afscan/example_19_part5_focus.mat"
    fscan = None
    single = None
    with h5py.File(fscan_path, "r") as data:
        fscan = data['sig']
        fscan = np.array(fscan)
    with h5py.File(single_path, "r") as data:
        single = data['sig']
        single = np.array(single)
    # fscan_estimator.time_frequency_estimate(fscan[13100, :])
    # single_estimator.time_frequency_estimate(single[14000, :])

    area = (int(1/(fscan_vr/PRF)), int(3/(c/(2*fs))))
    print("area: ", area)
    # fscan_estimator.dot_estimate(fscan[7700:8000, 8900:9105], area, 16)
    # single_estimator.dot_estimate(single[9300:9600, 4900:5100], (int(1/(single_vr/PRF)), int(3/(c/(2*fs)))), 16)
    fscan_bench = np.abs(fscan[14500, 14400])
    single_bench = np.abs(single[15800, 14700])

    print("shape of fscan: ", fscan.shape)
    print("shape of single: ", single.shape)
    fscan_power_mean = np.mean(np.abs(fscan), axis=0)
    single_power_mean = np.mean(np.abs(single), axis=0)
    fscan_snr = 20*np.log10(fscan_power_mean/fscan_bench)
    single_snr = 20*np.log10(single_power_mean/single_bench)

    H = 2922
    phi = np.rad2deg(61.83)
    Rc = 6371
    range_swath = (np.arange(fscan.shape[1])-fscan.shape[1]//2) * c / (2 * fs) + Rc
    doa = np.arccos(H/range_swath)
    ground_swath = H*np.tan(doa)
    ground_swath = ground_swath - ground_swath[0]
    
    plt.figure()
    plt.plot(ground_swath, fscan_snr, label="FSAR")
    plt.plot(ground_swath, single_snr, label="Conventional SAR")
    plt.xlabel("ground (m)")
    plt.ylabel("SCR (dB)")
    plt.legend()
    plt.grid()
    plt.savefig("../fig/afscan/snr.png", dpi=300)

    fscan_thresh = fscan_snr > 3
    single_thresh = single_snr > 3
    fscan_left = 0
    fscan_right = 0
    single_left = 0
    single_right = 0
    for i in range(len(fscan_thresh)-1):
        if fscan_left == 0 and fscan_thresh[i] == True:
            fscan_left = i
        if single_left == 0 and single_thresh[i] == True:
            single_left = i
    for i in range(len(fscan_thresh)-1, 0, -1):
        if fscan_right == 0 and fscan_thresh[i] == True:
            fscan_right = i
        if single_right == 0 and single_thresh[i] == True:
            single_right = i
    fscan_swath = ground_swath[fscan_right] - ground_swath[fscan_left]
    single_swath = ground_swath[single_right] - ground_swath[single_left]
    print("fscan swath: {} m".format(fscan_swath))
    print("single swath: {} m".format(single_swath))