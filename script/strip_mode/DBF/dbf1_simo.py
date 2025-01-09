import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

class DBF_Tradition:
    def __init__(self, La, N, fc, H, beta, Tp, Br):
        self.La = La
        self.N = N
        self.fc = fc
        self.H = H
        self.beta = beta
        self.Tp = Tp
        self.Br = Br
        self.c = 299792458
        self.Re = 6371.393e3
        self.lambda_ = self.c/fc
        self.d_ra = La/N
        self.Kr = self.Br/self.Tp
        self.Fr = 5*self.Br

    def ant_diagram(self):

        ## 子孔径相较于参考孔径的距离
        d = (cp.arange(0, self.N)-(self.N-1)/2)*self.d_ra
        # d = cp.linspace(-La/2, La/2, N)

        ## 下视角
        theta = cp.linspace(-20, 80, 10000)
        theta = cp.deg2rad(theta) 

        ## channel weights
        P_com = cp.zeros(len(theta), dtype=cp.complex128)
        P_sub = cp.abs(cp.sinc(self.d_ra*cp.sin(theta-self.beta)/self.lambda_))
        theta_s = self.beta+cp.deg2rad(-0.5)
        for i in range(self.N):
            w = cp.exp(-2j*cp.pi*d[i]*cp.sin(theta-self.beta)/self.lambda_)
            r = cp.exp(2j*cp.pi*d[i]*cp.sin(theta_s-self.beta)/self.lambda_)
            P_com += P_sub*w*r

        P_com = cp.abs(P_com)
        P_com = P_com/cp.max(P_com)
        P_com = 10*cp.log10(P_com)
        P_sub = P_sub/cp.max(P_sub)
        P_sub = 10*cp.log10(P_sub)
        plt.figure(1)
        plt.plot(cp.rad2deg(theta).get(), P_com.get(), label='P_com')
        plt.plot(cp.rad2deg(theta).get(), P_sub.get(), label='P_sub')
        plt.xlabel('theta')
        plt.ylabel('P')
        plt.legend(loc='best')
        plt.grid()
        plt.savefig("../../../fig/dbf/天线方向图.pdf", dpi=300)

    def fir_dbf_compare(self):

        point_x = cp.array([150e3, 250e3, 350e3])
        point_r = cp.sqrt(point_x**2+self.H**2)
        tau_min = cp.sqrt(150e3**2+self.H**2)*2/self.c
        tau_max = cp.sqrt(400e3**2+self.H**2)*2/self.c
        Nr = int(cp.ceil(((tau_max-tau_min))*self.Fr))

        tau =  point_r[1]*2/self.c + cp.arange(-Nr/2, Nr/2)*(1/self.Fr)
        f_tau = cp.fft.fftshift(cp.arange(-Nr/2, Nr/2)*self.Fr/Nr)
        ## no fir delay
        signal_r1 = cp.zeros(Nr, dtype=cp.complex128)

        ## fir delay
        signal_r2 = cp.zeros(Nr, dtype=cp.complex128)

        theta_tau = cp.arccos(((self.H+self.Re)**2+(tau*self.c/2)**2-self.Re**2)/(2*(self.H+self.Re)*(tau*self.c/2)))
        delta_theta = cp.gradient(theta_tau, tau)

        ## 这个的选取对最后结果影响比较大
        delta_theta_c = delta_theta[Nr-1]

        for j in range(len(point_r)):
            R = point_r[j]
            t0 = R*2/self.c
            theta = cp.arccos(((self.H+self.Re)**2+R**2-self.Re**2)/(2*(self.H+self.Re)*R))
            # P_sub = cp.abs(cp.sinc(self.d_ra*cp.sin(theta-self.beta)/self.lambda_))
            P_sub = 1
            for i in range(self.N):
                d = (i-(self.N-1)/2)*self.d_ra
                ti = t0-d*cp.sin(theta-self.beta)/self.c
                Wr = cp.abs(tau-ti)<self.Tp/2
                phase = cp.exp(1j*cp.pi*self.Kr*(tau-ti)**2)*cp.exp(-2j*cp.pi*self.fc*ti)
                channel_ri = P_sub*Wr*phase
                wi = cp.exp(-2j*cp.pi*d*cp.sin(theta-self.beta)/self.lambda_)
                rwi = channel_ri*wi
                signal_r1 += rwi
                delay = d*delta_theta_c/(self.lambda_*self.Kr)
                Hk = cp.exp(2j*cp.pi*delay*f_tau)
                signal_r2 += cp.fft.ifft(cp.fft.fft(rwi)*Hk)

        W = cp.sqrt((tau*self.c/2)**2 - self.H**2)/1e3
        plt.figure(2)
        plt.subplot(221)
        plt.plot(W.get(), cp.real(signal_r1).get())
        plt.xlabel('grand range')
        plt.ylabel('Amplitude')
        plt.title('no fir delay')
        plt.grid()

        plt.subplot(222)
        plt.plot(W.get(), cp.real(theta_tau).get())
        plt.xlabel('grand range')
        plt.ylabel('Amplitude')
        plt.title("fir delay")
        plt.grid()

        Hr = cp.exp(1j*cp.pi*f_tau**2/self.Kr)
        r1_compress = cp.abs(cp.fft.ifft(cp.fft.fft(signal_r1)*Hr))
        r2_compress = cp.abs(cp.fft.ifft(cp.fft.fft(signal_r2)*Hr))
        r1_compress = r1_compress/cp.max(r1_compress)
        r2_compress = r2_compress/cp.max(r2_compress)
        r1_compress = 10*cp.log10(r1_compress)
        r2_compress = 10*cp.log10(r2_compress)
        plt.subplot(223)
        plt.plot(W.get(), (r1_compress).get())
        plt.xlabel('grand range')
        plt.ylabel('Amplitude/db')
        plt.title("no fir delay")
        plt.grid()

        plt.subplot(224)
        plt.plot(W.get(), (r2_compress).get())
        plt.xlabel('grand range')
        plt.ylabel('Amplitude/db')
        plt.title("fir delay")
        plt.grid()

        plt.tight_layout()
        plt.savefig("../../../fig/dbf/通道时延对dbf信号的影响.pdf", dpi=300)

    
    def dbf_algo_improve(self):
        point_x = cp.array([150e3, 250e3, 350e3])
        point_r = cp.sqrt(point_x**2+self.H**2)
        tau_min = cp.sqrt(150e3**2+self.H**2)*2/self.c
        tau_max = cp.sqrt(400e3**2+self.H**2)*2/self.c
        Nr = int(cp.ceil(((tau_max-tau_min))*self.Fr))

        tc = point_r[1]*2/self.c
        tau = point_r[1]*2/self.c + cp.arange(-Nr/2, Nr/2)*(1/self.Fr)
        f_tau = cp.fft.fftshift(cp.arange(-Nr/2, Nr/2)*self.Fr/Nr)

        Hr = cp.exp(1j*cp.pi*f_tau**2/self.Kr)
        ## no process
        signal_r1 = cp.zeros(Nr, dtype=cp.complex128)

        ## bulk process compress before sum
        signal_r2 = cp.zeros(Nr, dtype=cp.complex128)

        ## bulk and residual process compress after sum
        signal_r3 = cp.zeros(Nr, dtype=cp.complex128)

        theta_tau = cp.arccos(((self.H+self.Re)**2+(tau*self.c/2)**2-self.Re**2)/(2*(self.H+self.Re)*(tau*self.c/2)))
        delta_theta = cp.gradient(theta_tau, tau)

        sin_theta_tau = cp.sin(theta_tau - beta)
        delta_sin_theta = cp.gradient(sin_theta_tau, tau)

        ## 这个的选取对最后结果影响比较大
        delta_theta_c = delta_theta[Nr/2]
        delta_sin_theta_c = delta_sin_theta[Nr/2]
        theta_c = cp.arccos(((self.H+self.Re)**2+(point_r[1])**2-self.Re**2)/(2*(self.H+self.Re)*(point_r[1])))

        for j in range(len(point_r)):
            R = point_r[j]
            t0 = R*2/self.c
            theta = cp.arccos(((self.H+self.Re)**2+R**2-self.Re**2)/(2*(self.H+self.Re)*R))
            # P_sub = cp.abs(cp.sinc(self.d_ra*cp.sin(theta-self.beta)/self.lambda_))
            P_sub = 1
            for i in range(self.N):
                d = (i-(self.N-1)/2)*self.d_ra
                ti = t0-d*cp.sin(theta-self.beta)/self.c
                Wr = cp.abs(tau-ti)<self.Tp/2
                phase = cp.exp(1j*cp.pi*self.Kr*(tau-ti)**2)*cp.exp(-2j*cp.pi*self.fc*ti)
                channel_ri = P_sub*Wr*phase
                wi = cp.exp(-2j*cp.pi*d*cp.sin(theta-self.beta)/self.lambda_)
                delay_fir = d*delta_theta_c/(self.lambda_*self.Kr)
                ## fir delay
                Hk = cp.exp(2j*cp.pi*delay_fir*f_tau)
                channel_r1i = cp.fft.ifft(cp.fft.fft(channel_ri*wi)*Hk)
                signal_r1 += channel_r1i

                ## bulk process, no fir delay
                delay_bulk = d*cp.sin(theta_c-beta)/self.c
                H_delay1 = cp.exp(-2j*cp.pi*delay_bulk*f_tau)
                channel_rd1 = cp.fft.ifft(cp.fft.fft(channel_ri)*H_delay1)
                ## compress
                signal_r2 += cp.fft.ifft(cp.fft.fft(channel_rd1)*Hr)*wi
                ## residual process
                ## 变标方程
                Fsc = cp.exp(-1j*cp.pi*self.Kr*d*delta_theta_c*(tau-tc)**2/self.c)
                channel_rsc = channel_rd1*Fsc
                ## compress
                channel_rsc = cp.fft.ifft(cp.fft.fft(channel_rsc)*Hr)
                sc_factor = -d*delta_sin_theta_c/self.c
                wi_improve = wi*cp.exp(-1j*self.Kr*sc_factor*(tau-tc)**2)
                signal_r3 += channel_rsc*wi_improve

        signal_r1 = cp.fft.ifft(cp.fft.fft(signal_r1)*Hr)
        signal_r1 = signal_r1/(cp.max(cp.abs(signal_r1)))
        signal_r2 = signal_r2/(cp.max(cp.abs(signal_r2)))
        signal_r3 = signal_r3/(cp.max(cp.abs(signal_r3)))
        upsample_factor = 10
        slice_size = 100
        indices = cp.linspace(0, slice_size-1, slice_size)
        up_indices = cp.linspace(0, slice_size-1, slice_size*upsample_factor)
        up_tau = cp.arange(-slice_size*upsample_factor/2, slice_size*upsample_factor/2)*(1/self.Fr/upsample_factor)
        up_R = up_tau*self.c/2
        plt.figure(figsize=(10, 10))
        for i in range(3):
            R = point_r[i]
            t0 = R*2/self.c
            pos = cp.round((t0 - (point_r[1]*2/self.c - Nr/2*(1/self.Fr)))/((1/self.Fr))).astype(int) 
            data_1 = cp.abs(cp.interp(up_indices, indices, signal_r1[pos-slice_size/2:pos+slice_size/2]))
            data_2 = cp.abs(cp.interp(up_indices, indices, signal_r2[pos-slice_size/2:pos+slice_size/2]))
            data_3 = cp.abs(cp.interp(up_indices, indices, signal_r3[pos-slice_size/2:pos+slice_size/2]))
            data_1 = 10*cp.log10(data_1)
            data_2 = 10*cp.log10(data_2)
            data_3 = 10*cp.log10(data_3)
            x = up_R+point_r[i]
            x = cp.sqrt(x**2-self.H**2)/1e3
            plt.subplot(3, 3, i*3+1)
            plt.plot(x.get(), (data_1).get())
            plt.title('1 R = %d km'%(point_r[i]/1e3))
            plt.grid()
            plt.ylim(-20, 0)
            plt.subplot(3, 3, i*3+2)
            plt.plot(x.get(), (data_2).get())
            plt.title('2 R = %d km'%(point_r[i]/1e3))
            plt.ylim(-20, 0)
            plt.grid()
            plt.subplot(3, 3, i*3+3)
            plt.plot(x.get(), (data_3).get())
            plt.title('3 R = %d km'%(point_r[i]/1e3))
            plt.grid()
            plt.ylim(-20, 0)
        plt.tight_layout()
        plt.savefig("../../../fig/dbf/改进算法对dbf信号的影响.pdf", dpi=900)

    def dbf_nesz(self, F, PRF, Loss, T, P, Vs, theta_s, Loss_az, Laz):
    # L:天线效率
    # T:等效噪声温度     
    # P:峰值功率
    # Vs:雷达速度
    # theta_s:斜视角
        k = 1.38e-23 # 玻尔兹曼常数
        theta = cp.deg2rad(cp.linspace(28, 34, 10000))
        theta_HRe = cp.arcsin((self.H+self.Re)*cp.sin(theta)/self.Re)
        theta_R = theta_HRe - theta
        R0 = self.Re*cp.sin(theta_R)/cp.sin(theta)
        # R0 = self.H/cp.cos(theta)
        R = R0/cp.cos(theta_s)
        incident = cp.arccos(R/self.Re)
        G_t = 4*cp.pi*self.d_ra*Laz*cp.sinc(self.d_ra*cp.sin(theta-self.beta)/self.lambda_)**2/(self.lambda_**2)
        G_r = G_t
        nesz_single = 256*cp.pi**3*R0**3*Vs*cp.sin(incident)*k*T*self.Br*Loss*F*Loss_az/(P*PRF*self.Tp*self.lambda_**3*self.c*G_r*G_t)
        nesz = nesz_single/self.N
        nesz = 10*cp.log10(nesz)
        plt.figure()
        plt.plot(cp.rad2deg(theta).get(), nesz.get())
        plt.xlabel('theta')
        plt.ylabel('nesz/db')
        plt.title('NESZ')
        plt.grid()
        plt.savefig("../../../fig/dbf/NESZ.pdf", dpi=300)
        pass

if __name__ == '__main__':
    La = 2
    N = 15
    fc = 9.6e9
    H = 700e3
    beta = cp.deg2rad(32.5)   ## 法线下视角
    Tp = 40e-6
    Br = 100e6
    DBF_sim = DBF_Tradition(La, N, fc, H, beta, Tp, Br)
    # DBF_sim.ant_diagram()
    # DBF_sim.fir_dbf_compare()
    # DBF_sim.dbf_algo_improve()
    Laz = 12
    Pt = 1e4
    Loss = 0.78
    Loss_az = 1
    F = 1
    PRF = 1500
    T = 290
    Vs = 7200
    theta_s = cp.deg2rad(0)
    # DBF_sim.dbf_nesz(F, PRF, Loss, T, Pt, Vs, theta_s, Loss_az, Laz)
    theta = cp.array([10, 15.89])
    theta = cp.deg2rad(theta)
    theta_HRe = cp.arcsin((H+DBF_sim.Re)*cp.sin(theta)/DBF_sim.Re)
    theta_R = theta_HRe - theta
    R0 = DBF_sim.Re*cp.sin(theta_R)/cp.sin(theta)
    R_w = R0[1]-R0[0]
    tau = R_w*2/DBF_sim.c
    rate = 450*2*1.2*8*3
    rate_mean = rate*tau*1288
    print(rate_mean)