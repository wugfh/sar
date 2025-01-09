import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

class DBF_MIMO:
    kernel_code = '''
    extern "C" 
    #define M_PI 3.14159265358979323846
    __global__ void sinc_interpolation(
        const double* echo_ftau_feta,
        const int* delta_int,
        const double* delta_remain,
        double* echo_ftau_feta_stolt,
        int Na, int Nr, int sinc_N) {

        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        
        if (i < Na && j < Nr) {
            int  del_int = delta_int[i * Nr + j];
            double del_remain = delta_remain[i * Nr + j];
            double predict_value = 0;
            double sum_sinc = 0;
            for (int m = 0; m < sinc_N; ++m) {
                double sinc_x = del_remain - (m - sinc_N/2);
                double sinc_y = sin(M_PI * sinc_x) / (M_PI * sinc_x);
                if(sinc_x < 1e-6 && sinc_x > -1e-6) {
                    sinc_y = 1;
                }
                int index = del_int + j + m - sinc_N/2;
                sum_sinc += sinc_y;
                if (index >= Nr) {
                    predict_value += 0;
                } else if (index < 0) {
                    predict_value += 0;
                } else {
                    predict_value += echo_ftau_feta[i * Nr + index] * sinc_y;
                }
            }
            echo_ftau_feta_stolt[i * Nr + j] = predict_value/sum_sinc;
        }
    }
    '''
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
        echo = cp.zeros(self.azchan_N, self.Na, self.Nr)
        for i in range(self.rchan_N):
            dr = self.dra*(i-(self.rchan_N-1)/2)
            for j in range(self.azchan_N):
                da = self.dza*(i-(self.azchan_N-1)/2)
                for k in range(len(point_r)):
                    R_eta_tx = cp.sqrt(point_r[k]**2+(self.Vr*mat_eta - point_a[k])**2)
                    R_eta_rx = cp.sqrt(point_r[k]**2+(self.Vr*(mat_eta-da/self.Vr) - point_a[k])**2)
                    R_eta = (R_eta_rx+R_eta_tx)/2
                    pulze_width = 0.886*(point_r[i]/cp.cos(self.theta_rc))*self.lambda_/(self.dza*self.Vr*cp.cos(self.theta_rc))
                    Wa = cp.abs(mat_eta - eta_c - (point_a[k])/self.Vr) < pulze_width

                    # MIMO 模式下，发射端会同时发射n个不同信号
                    ## 发射信号1
                    phi_tar = cp.arccos(point_r[i]/self.H)
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
        f_tau = cp.fft.fftshift(cp.arange(-self.Nr/2, self.Nr/2)*(self.Fr/self.Nr))
        echo_rc = cp.zeros(cp.shape(echo))
        mat_f_tau = cp.ones(self.Na,1) @ f_tau
        Hr = cp.exp(1j*cp.pi*mat_f_tau**2/self.Kr)
        for i in range(self.azchan_N):
            echo_chan = cp.squeeze(echo[i, : ,:])
            echo_rcompress = cp.fft.ifft(cp.fft.fft(echo_chan)*Hr)
            echo_rc[i,:,:] = echo_rcompress

        return echo_rc    

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
        pass

    def azimuth_reconstruct(self, echo_rc):
        P_matrix = self.reconstruct_filter()
        self.Na_up = self.Na*self.azchan_N
        f_eta = 2*self.Vr*cp.sin(self.theta_rc)/self.lambda_ + cp.fft.fftshift(cp.arange(-self.Na_up/2, self.Na_up/2)*(self.PRF/self.Na))
        mat_f_eta = cp.transpose(f_eta) @ cp.ones(1, self.Nr)

        out_band = cp.zeros((self.azchan_N, self.Na, self.Nr), dtype=cp.complex128) 
        for i in range(self.azchan_N):
            aperture = cp.squeeze(echo_rc[i, :, :])
            P_aperture = cp.squeeze(P_matrix[i, :, :])
            tmp = aperture @ P_aperture
            for j in range(self.azchan_N):
                out_band[j, i, :] = tmp[:, j]

        echo_f_rc  = cp.zeros(self.Na_up, self.Nr)        
        for j in range(self.azchan_N):
            echo_f_rc[j * self.Na: (j + 1) * self.Na, :] = cp.fft.fftshift(cp.squeeze(out_band[j, :, :]), axes=0)
