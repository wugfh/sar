import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append(r"../")
from sinc_interpolation import SincInterpolation

class DBF_MIMO:

    def __init__(self, fc, H, Br, Tp, Lar, rN, Laz, aN, Ba, theta_rc, Ta, tN, phi):
        self.fc = fc
        self.H = H
        self.Br = Br
        self.Fr = 1.2*Br
        self.Tp = Tp
        self.Lar = Lar
        self.rchan_N = rN
        self.Laz = Laz
        self.azchan_N = aN
        self.Ba = Ba
        self.theta_rc = theta_rc ## 斜视角
        self.Ta = Ta
        self.tN = tN
        self.phi = phi
        self.c = 299792458
        self.Vr = Ba*Laz/(aN*2*cp.cos(theta_rc))
        self.PRF = 1.5*(Ba/(tN*aN-(tN-1)))
        self.Na = int(cp.ceil(cp.ceil(self.PRF*Ta)/2)*2)
        self.Nr = int(cp.ceil(cp.ceil(self.Fr*Tp)/2)*2)
        self.dra = self.Lar/self.rchan_N
        self.dza = self.Laz/self.azchan_N
        self.Rc = (self.H/cp.cos(phi))/cp.cos(self.theta_rc)
        self.lambda_ = self.c/self.fc
        self.Kr = Br/Tp
    
    def get_sim_echo(self):
        tau = 2*self.Rc/self.c + cp.arange(-self.Nr/2, self.Nr/2, 1)*(1/self.Fr)
        eta_c = -self.Rc*cp.sin(self.theta_rc)/self.c
        eta = eta_c + cp.arange(-self.Na/2, self.Na/2, 1)*(1/self.PRF)
        mat_tau, mat_eta = cp.meshgrid(tau, eta)
        point_r = self.H/cp.cos(self.phi)+cp.linspace(-10000, 10000, 5)
        point_a = cp.linspace(-10000, 10000, 5)
        echo = cp.zeros((self.azchan_N, self.Na, self.Nr), dtype=cp.complex128)
        for i in range(self.rchan_N):
            dr = self.dra*(i-(self.rchan_N-1)/2)
            for j in range(self.azchan_N):
                da = self.dza*(i-(self.azchan_N-1)/2)
                for k in range(len(point_r)):
                    R_eta_tx = cp.sqrt(point_r[k]**2+(self.Vr*mat_eta - point_a[k])**2)
                    R_eta_rx = cp.sqrt(point_r[k]**2+(self.Vr*(mat_eta-da/self.Vr) - point_a[k])**2)
                    R_eta = (R_eta_rx+R_eta_tx)/2
                    pulze_width = 0.886*(point_r[k]/cp.cos(self.theta_rc))*self.lambda_/(self.dza*self.Vr*cp.cos(self.theta_rc))
                    Wa = cp.abs(mat_eta - eta_c - (point_a[k])/self.Vr) < pulze_width

                    # MIMO 模式下，发射端会同时发射n个不同信号
                    ## 发射信号1
                    phi_tar = cp.arccos(point_r[k]/self.H)
                    mat_tau_ichan = mat_tau - da*cp.sin(phi_tar-self.phi)/self.c 
                    Wr = cp.abs(mat_tau_ichan - 2*R_eta/self.c) < self.Tp/2
                    Phase = cp.exp(1j*cp.pi*self.Kr*(mat_tau_ichan - 2*R_eta/self.c)**2)*cp.exp(-2j*cp.pi*self.fc*(2*R_eta/self.c))
                    echo1 = Wr*Wa*Phase*cp.exp(2j*cp.pi*(dr)/self.lambda_)
                    ## 发射信号2
                    Wr1 = cp.abs(mat_tau_ichan - 2*R_eta/self.c - self.Tp/4) < self.Tp/4
                    Phase1 = cp.exp(1j*cp.pi*self.Kr*(mat_tau_ichan - 2*R_eta/self.c - self.Tp/2)**2)*cp.exp(-2j*cp.pi*self.fc*(2*R_eta/self.c))
                    Wr2 = cp.abs(mat_tau_ichan - 2*R_eta/self.c + self.Tp/4) < self.Tp/4
                    Phase2 = cp.exp(1j*cp.pi*self.Kr*(mat_tau_ichan - 2*R_eta/self.c + self.Tp/2)**2)*cp.exp(-2j*cp.pi*self.fc*(2*R_eta/self.c))
                    echo2 = Wa*(Wr1*Phase1+Wr2*Phase2)

                    echo[j,:,:] = echo1+echo2
        return echo

    def range_compress(self, echo):
        f_tau = cp.fft.fftshift(cp.arange(-self.Nr/2, self.Nr/2)*(self.Fr/self.Nr))[cp.newaxis, :]
        echo_f_rc = cp.zeros((self.Na, self.Nr, self.azchan_N), dtype=cp.complex128)

        mat_f_tau = cp.ones((self.Na,1)) @ f_tau
        Hr = cp.exp(1j*cp.pi*mat_f_tau**2/self.Kr)
        for i in range(self.azchan_N):
            echo_chan = cp.squeeze(echo[i, : ,:])
            echo_rcompress = cp.fft.ifft(cp.fft.fft(echo_chan)*Hr)
            echo_f_rc[:,:,i] = cp.fft.fft(echo_rcompress, self.Na, axis=0)

        return echo_f_rc    

    def reconstruct_filter(self):
        P_matrix = cp.zeros((self.Na, self.azchan_N, self.azchan_N), dtype=cp.complex128)
        H_matrix = cp.zeros((self.Na, self.azchan_N, self.azchan_N), dtype=cp.complex128)
        # 计算方位向响应
        f_eta = 2*self.Vr*cp.sin(self.theta_rc)/self.lambda_ + cp.fft.fftshift(cp.arange(-self.Na/2, self.Na/2)*(self.PRF/self.Na))
        for k in range(self.azchan_N):
            for n in range(self.azchan_N):
                H_matrix[:, k, n] = cp.exp(- 1j * cp.pi * (n * self.dza) / self.Vr * (f_eta + (k-(self.azchan_N-1)/2) * self.PRF))
        for j in range(self.Na): 
            P_matrix[j] = cp.linalg.inv(H_matrix[j])    
        
        return P_matrix
    
    def rd_rcmc(self, echo_f_rc):
        tau = (2*self.Rc/self.c + cp.arange(-self.Nr/2, self.Nr/2, 1)*(1/self.Fr))[:, cp.newaxis]
        mat_tau = cp.ones((self.Na_up ,1)) @ cp.transpose(tau)
        f_eta_up = 2*self.Vr*cp.sin(self.theta_rc)/self.lambda_ + cp.fft.fftshift(cp.arange(-self.Na_up/2, self.Na_up/2)*(self.PRF/self.Na))[:, cp.newaxis]
        mat_f_eta = (f_eta_up) @ cp.ones((1, self.Nr))
        mat_R = mat_tau*self.c/2
        delta_R = self.lambda_**2 * mat_f_eta**2 * mat_R/(8*self.Vr)
        delta = (delta_R*2/self.c)/(1/self.Fr)
        sinc_intp = SincInterpolation()
        echo_f_rcmc = sinc_intp.sinc_interpolation(echo_f_rc, delta, self.Na_up, self.Nr, 8)
        return echo_f_rcmc

    def azimuth_reconstruct(self, echo_f_rc):
        P_matrix = self.reconstruct_filter()
        self.Na_up = self.Na*self.azchan_N
        out_band = cp.zeros((self.azchan_N, self.Na, self.Nr), dtype=cp.complex128) 
        for i in range(self.Na):
            aperture = cp.squeeze(echo_f_rc[i, :, :])
            P_aperture = cp.squeeze(P_matrix[i, :, :])
            tmp = aperture @ P_aperture
            for j in range(self.azchan_N):
                out_band[j, i, :] = tmp[:, j]

        echo_f_rc  = cp.zeros((self.Na_up, self.Nr), dtype=cp.complex128)        
        for j in range(self.azchan_N):
            echo_f_rc[j * self.Na: (j + 1) * self.Na, :] = cp.fft.fftshift(cp.squeeze(out_band[j, :, :]), axes=0)

        return echo_f_rc
        
    def azimuth_compress(self, echo_f_rc):
        echo_f_rc = self.rd_rcmc(echo_f_rc)
        tau = 2*self.Rc/self.c + (cp.arange(-self.Nr/2, self.Nr/2, 1)*(1/self.Fr))[cp.newaxis, :]
        mat_tau = cp.ones((self.Na_up ,1)) @ tau
        mat_R = mat_tau*self.c/2
        Ka = 2 * self.Vr**2 * cp.cos(self.theta_rc)**2 / (self.lambda_ * mat_R)
        f_eta_up = 2*self.Vr*cp.sin(self.theta_rc)/self.lambda_ + cp.fft.fftshift(cp.arange(-self.Na_up/2, self.Na_up/2)*(self.PRF/self.Na))[:, cp.newaxis]
        mat_f_eta = (f_eta_up) @ cp.ones((1, self.Nr))
        Ha = cp.exp(-1j*cp.pi*mat_f_eta**2/Ka)
        out = cp.fft.ifft(Ha*echo_f_rc, self.Na_up, axis=0)

        return out



if __name__ == '__main__':
    cp.cuda.Device(1).use()
    fc = 9.6e9
    H = 700e3
    Br = 100e6
    Tp = 20e-6
    Lar = 2.5
    rN = 25
    Laz = 12
    aN = 3
    Ba = 3754
    theta_rc = cp.deg2rad(0)
    Ta = 1
    tN = 2
    phi = cp.deg2rad(30.48)
    dbf_mimo = DBF_MIMO(fc, H, Br, Tp, Lar, rN, Laz, aN, Ba, theta_rc, Ta, tN, phi)
    echo = dbf_mimo.get_sim_echo()
    echo_f_rc = dbf_mimo.range_compress(echo)
    echo_f_rc = dbf_mimo.azimuth_reconstruct(echo_f_rc)
    out = dbf_mimo.azimuth_compress(echo_f_rc)
    
    plt.figure(figsize = (8,8))
    plt.imshow((cp.abs(echo_f_rc)).get(), cmap='jet', aspect='auto')
    plt.colorbar()
    plt.savefig("../../../fig/dbf/dbf_mimo_rd.png")

    plt.figure(figsize = (8,8))
    plt.imshow((cp.abs(out)).get(), cmap='jet', aspect='auto')
    plt.colorbar()
    plt.savefig("../../../fig/dbf/dbf_mimo_out.png")


