import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import cv2

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
            fscan_rtarget = np.max(np.abs(target), axis=0)
            fscan_rtarget = fscan_rtarget/np.max(fscan_rtarget)
            x_dr = np.linspace(dr[0], dr[1], len(fscan_rtarget))

            plt.subplot(3, self.points_n, self.points_n + cnt+1)
            plt.plot(x_dr, 20*np.log10(fscan_rtarget))
            plt.grid()
            plt.ylim(-30, 0)
            plt.xlabel("range(m)")
            plt.ylabel("amplitude(dB)")
            plt.title("({}{})".format(letter_mapping[(self.points_n + cnt+1)%26],1))

            fscan_azimuth_res = self.get_azimuth_IRW(np.abs(target), uprate)
            fscan_atarget = np.max(np.abs(target), axis=1)
            fscan_atarget = fscan_atarget/np.max(fscan_atarget)
            x_da = np.linspace(da[0], da[1], len(fscan_atarget))

            plt.subplot(3, self.points_n, 2*self.points_n + cnt+1)
            plt.plot(x_da, 20*np.log10(fscan_atarget))
            plt.grid()
            plt.ylim(-30, 0)
            plt.xlabel("azimuth(m)")
            plt.ylabel("amplitude(dB)")
            plt.title("({}{})".format(letter_mapping[(self.points_n + cnt+1)%26],2))

            image_show = np.abs(target_up)/np.max(np.max(np.abs(target_up)))
            image_show = 20*np.log10(image_show)            
            plt.subplot(3, self.points_n, cnt+1)
            # plt.imshow(np.abs(tmp), aspect="auto", cmap='jet', extent=[dr[0], dr[1], da[0], da[1]])
            plt.imshow(image_show, aspect="auto", cmap='jet', extent=[dr[0], dr[1], da[0], da[1]], vmax = 0, vmin = -60)
            plt.ylabel("azimuth(m)")
            plt.xlabel("range(m)")
            colorbar = plt.colorbar()
            colorbar.ax.set_title("dB")
            plt.title("({}{})".format(letter_mapping[(cnt+1)%26],0))


            print("range irw: ", fscan_range_res)
            print("azimuth irw: ", fscan_azimuth_res)
            print("range pslr: ", self.get_pslr(fscan_rtarget))
            print("azimuth pslr: ", self.get_pslr(fscan_atarget))
            # print("range islr: ", self.get_islr(fscan_rtarget))
            # print("azimuth islr: ", self.get_islr(fscan_atarget))

            image_copy[max_index[0]-area[0]//2:max_index[0]+area[0]//2, max_index[1]-area[1]//2:max_index[1]+area[1]//2] = 0
            cnt = cnt+1
        
        plt.tight_layout()

        plt.savefig(self.path+"dot_estimate.png", dpi=300)

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