import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.optimize as optimize
from multiprocessing import Process, Queue

from matplotlib import font_manager

my_font = font_manager.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


class SlideSpotDesign:
    def __init__(self):
        self.H = 280e3
        self.c = 299792458 # Speed of light in m/s
        self.EarthMass = 5.972e24 # kg
        self.Re = 6371e3
        self.Gravitational = 6.67430e-11
        self.Vs = np.sqrt(self.Gravitational*self.EarthMass/(self.Re + self.H))  
        self.Vg = self.Vs * self.Re / (self.Re + self.H)
        self.Vr = np.sqrt(self.Vg*self.Vs)

        self.da = 0.1   ## 方位向地距分辨率
        self.dg = 0.1   ## 距离向地距分辨率
        self.f0 = 35e9  ## 载波频率
        self.Tp = 40e-6 ## 脉冲宽度
        self.groud_extent = 4e3
        self.azimuth_extent = 4e3

        self.lambda_ = self.c / self.f0

        # self.beta = np.deg2rad(25) ## 下视角中心
        # self.look_angle_left, self.look_angle_right = self.calculate_scanwidth(self.groud_extent) ## 下视角范围
        self.beta = np.deg2rad(np.arange(20, 45, 0.6)) ## 下视角中心
        look_angle_left = []
        look_angle_right = []
        for beta in self.beta:
            left, right = self.calculate_scanwidth(beta, self.groud_extent)
            look_angle_left.append(left)
            look_angle_right.append(right)
        self.look_angle_left = np.array(look_angle_left) ## 下视角范围左侧
        self.look_angle_right = np.array(look_angle_right)

        # self.Br = self.c/(2*self.dg * np.sin(self.beta)) ## 距离向带宽
        self.Br = 3.2e9


        tmp_angle = np.arcsin((self.H+self.Re)*np.sin(self.beta)/self.Re)
        tmp_angle = tmp_angle - self.beta
        self.R0 = self.Re*np.sin(tmp_angle)/np.sin(self.beta)
        PRF1 = np.array([8300])
        PRF2 = np.linspace(7900, 7300, 11)
        PRF3 = np.array([7000, 8200])
        PRF4 = np.linspace(7900, 7300, 8)
        PRF5 = np.linspace(7900, 7300, 8)
        PRF6 = np.linspace(7800, 7400, 6)
        PRF7 = np.linspace(7800, 7400, 4)
        PRF8 = np.linspace(7800, 7600, 2)
        self.PRF = np.concatenate([PRF1, PRF2, PRF3, PRF4, PRF5, PRF6, PRF7, PRF8])
        self.NB = 1

        self.A = 0.07
        self.Vf = self.Vg*self.A
        self.Ta = self.azimuth_extent/self.Vf
        self.Rtot = self.R0/(1-self.A)
        self.omega = self.Vg / self.Rtot
        self.psi_0 = 0
        self.psi_start = self.psi_0 - self.omega * self.Ta/2
        self.psi_end = self.psi_0 + self.omega * self.Ta/2
        self.theta_a = self.lambda_ * np.cos(self.psi_0)/(2*self.da/self.A)
        self.theta_c = 0
        self.La = 0.886*self.lambda_/self.theta_a
        self.Lr = 0.886*self.lambda_/(self.look_angle_right-self.look_angle_left)
        self.Lr = 0.75
        self.Bfov = self.Bfov_func(self.theta_a, 0)

        self.K = 1.38e-23                           #玻尔兹曼常数
        self.T = 320                                #温度
        self.Ln = 2.5                              ## 总体系统损耗
        
    def calculate_R0(self, look_angle):
        ## 星载
        if self.H > 100e3:
            tmp_angle = np.arcsin((self.H+self.Re)*np.sin(look_angle)/self.Re)
            tmp_angle = tmp_angle - look_angle
            R0 = self.Re*np.sin(tmp_angle)/np.sin(look_angle)
        ## 机载
        else:
            R0 = self.H/np.cos(look_angle)
        return R0
    
    def calculate_incident(self, R_eta):
        if self.H > 100e3:
            incident = np.arccos((self.Re**2+R_eta**2-(self.H+self.Re)**2)/(2*self.Re*R_eta))
            incident = np.pi - incident
        else:
            incident = np.arccos(self.H/R_eta)
        return incident
    
    
    def calculate_doa(self, R0):
        if self.H > 100e3:
            incident = np.arccos((self.Re**2+R0**2-(self.H+self.Re)**2)/(2*self.Re*R0)*(R0>self.H))
            doa = np.arcsin(self.Re*np.sin(incident)/(self.H+self.Re))
        else:
            doa = np.arccos(self.H/R0)
        return doa
    
    def calulate_extent(self, psi_start, psi_end, theta_a):
        dstart = self.R0 * np.tan(psi_start)
        dend = self.R0 * np.tan(psi_end)
        dstart_1 = self.R0*(np.tan(psi_start-theta_a/2)) - dstart
        dstart_2 = self.R0*(np.tan(psi_start+theta_a/2)) - dstart
        dend_1 = self.R0*(np.tan(psi_end-theta_a/2)) - dend
        dend_2 = self.R0*(np.tan(psi_end+theta_a/2)) - dend
        if dend_2 < dstart_1 or dend_1 > dstart_2:
            print("Error: The extent of the beam does not overlap.")
            return 0
        return min(dstart_2, dend_2) - max(dstart_1, dend_1)
    
    def calculate_scanwidth(self, beta, ground_width):
        ground_angle = ground_width/self.Re

        def func(x):
            g_left = np.arcsin((self.H+self.Re)/(self.Re/np.sin(beta-x))) - (beta-x)
            g_right = np.arcsin((self.H+self.Re)/(self.Re/np.sin(beta+x))) - (beta+x)
            return g_right-g_left-ground_angle
        
        equation = lambda x: func(x)

        solu = optimize.fsolve(equation, 0)

        look_angle_left = beta-solu[0]
        look_angle_right = beta+solu[0]

        return look_angle_left, look_angle_right
    
    def Bd_func(self, psi_start, psi_end, theta_a):
            return 2*self.Vs*np.abs(np.sin(psi_start-theta_a/2) - np.sin(psi_end+theta_a/2)) / (self.lambda_)
        
    def Bfov_func(self, theta_a, psi_start):
            return 2*self.Vs*np.abs(np.sin(psi_start+theta_a/2) - np.sin(psi_start-theta_a/2))/self.lambda_
        
    
    def get_spot_max_extent(self):


        ## 目标总的多普勒带宽大于某个值
        def constraint1(x):
            psi_start, psi_end, theta_a = x
            psi_start = np.deg2rad(psi_start)
            psi_end = np.deg2rad(psi_end)
            return self.Bd_func(psi_start, psi_end, theta_a)-self.Bd 
        
        def constraint2(x):
            psi_start, psi_end, theta_a = x
            psi_start = np.deg2rad(psi_start)
            psi_end = np.deg2rad(psi_end)
            theta_a = np.deg2rad(theta_a)
            return self.calulate_extent(psi_start, psi_end, theta_a)-self.azimuth_extent

        ## 求解最小PRF
        def objective(x):
            psi_start, psi_end, theta_a = x
            psi_start = np.deg2rad(psi_start)
            psi_end = np.deg2rad(psi_end)
            theta_a = np.deg2rad(theta_a)
            if psi_start*psi_end < 0:
                return self.Bfov_func(theta_a, 0)*1.1
            else:
                return self.Bfov_func(theta_a, np.sign(psi_start)*min(np.abs(psi_start), np.abs(psi_end)))*1.1

        bounds = [
            ((-10), (10)),  # psi_start
            ((-10), (10)),  # psi_end
            ((0.01), (10)) # theta_a
        ]

        cons = [
            {'type': 'ineq', 'fun': constraint1},
            {'type': 'ineq', 'fun': constraint2}
        ]

        def cons_for_evolution(x):
            return constraint1(x), constraint2(x)
        
        lb = [0, 0]
        ub = [0.2*self.Bd, 0.2*self.azimuth_extent]
        nlc = optimize.NonlinearConstraint(cons_for_evolution, lb, ub, jac='2-point')

        result = optimize.differential_evolution(
            objective,
            bounds=bounds,
            constraints=nlc,
            strategy='best1bin',
            maxiter=1000,
            popsize=15,
            tol=1e-6,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=None,
            polish=True,
            disp=False,
            updating='deferred'
        )
        res = result
        
        # x0 = [(0), (0), (0.3)]
        # res = optimize.minimize(objective, x0, bounds=bounds, constraints=cons, method="SLSQP")


        if res.success:
            min_PRF = res.fun
            print("最小PRF: {:.2f}".format(min_PRF))

            print("azimuth extent: {:.2f} m".format(self.calulate_extent(np.deg2rad(res.x[0]), np.deg2rad(res.x[1]), np.deg2rad(res.x[2]))))

            print("Bd: {:.2f} Hz".format(self.Bd_func(np.deg2rad(res.x[0]), np.deg2rad(res.x[1]), np.deg2rad(res.x[2]))))

            print("psi_start: {:.2f}°, psi_end: {:.2f}°, theta_a: {:.2f}°".format(
                (res.x[0]), (res.x[1]), (res.x[2])))
            self.psi_start = np.deg2rad(res.x[0])
            self.psi_end = np.deg2rad(res.x[1])
            self.theta_a = np.deg2rad(res.x[2])
            return min_PRF
        else:
            print("优化失败:", res.message)
            print("当前结果:")
            print("PRF: {:.2f}, azimuth extent {:.2f} m".format(res.fun, self.calulate_extent(np.deg2rad(res.x[0]), np.deg2rad(res.x[1]), np.deg2rad(res.x[2]))))
            print("psi_start: {:.2f}°, psi_end: {:.2f}°, theta_a: {:.2f}°".format(
                (res.x[0]), (res.x[1]), (res.x[2])))
            return None
            
    
    def zebra_diagram(self, prf, tau_rp):

        frac_min = (tau_rp + self.Tp) * prf  # 发射约束
        frac_max = -tau_rp * prf

        int_min = int(np.min(np.floor(2 * self.H * prf / self.c)))
        int_max = int(np.max(np.ceil(2 * np.sqrt((self.Re+self.H)**2 - (self.Re)**2) * prf / self.c)))

        plt.figure("PRF selection")

        # 遍历尽可能多的干扰
        for i in range(int_max + 1):
            R1 = (i + frac_min) * self.c / (2 * prf)
            gamma_cos = (R1**2 + (self.Re+self.H)**2 - self.Re**2) / (2 * R1 * (self.Re+self.H))
            # gamma_cos = self.H/R1
            gamma_cos = np.clip(gamma_cos, -1, 1)
            gamma = np.arccos(np.abs(gamma_cos))
            # belta = np.arcsin(R1 * np.sin(gamma) / self.Re)
            gamma1 = np.rad2deg(gamma)

            plt.plot(prf, gamma1, 'b')

            Rn = (i + frac_max) * self.c / (2 * prf)
            gamma_cos = (Rn**2 + (self.Re+self.H)**2 - self.Re**2) / (2 * Rn * (self.Re+self.H))
            # gamma_cos = self.H/Rn
            gamma_cos = np.clip(gamma_cos, -1, 1)
            gamma = np.arccos(np.abs(gamma_cos))
            # belta = np.arcsin(Rn * np.sin(gamma) / self.Re)
            gamma2 = np.rad2deg(gamma)

            plt.plot(prf, gamma2, 'b')
            plt.fill_between(prf, gamma1, gamma2, alpha=0.5, color='b')

        # 星下点干扰
        for i in range(10):
            R = (2 * self.H / self.c + i / prf - self.Tp/2) * self.c / 2
            gamma_cos = (R**2 + (self.Re+self.H)**2 - self.Re**2) / (2 * R * (self.Re+self.H))
            # gamma_cos = self.H/R
            gamma_cos = np.clip(gamma_cos, -1, 1)
            gamma = np.arccos(np.abs(gamma_cos))
            # belta = np.arcsin(R * np.sin(gamma) / self.Re)
            gamma1 = np.rad2deg(gamma)

            plt.plot(prf, gamma1, 'r')

            R = (2 * self.H / self.c + i / prf ) * self.c / 2
            gamma_cos = (R**2 + (self.Re+self.H)**2 - self.Re**2) / (2 * R * (self.Re+self.H))
            # gamma_cos = self.H/R
            gamma_cos = np.clip(gamma_cos, -1, 1)
            gamma = np.arccos(np.abs(gamma_cos))
            # belta = np.arcsin(R * np.sin(gamma) / self.Re)
            gamma2 = np.rad2deg(gamma)
            
            plt.plot(prf, gamma2, 'r')
            plt.fill_between(prf, gamma1, gamma2, alpha=0.5, color='r')
        
        for i in range(len(self.PRF)):
            plt.vlines(self.PRF[i], ymin=np.rad2deg(self.look_angle_left[i]), ymax=np.rad2deg(self.look_angle_right[i]), color='k')
    
    
        plt.grid()
        plt.xlabel("PRF/Hz")
        plt.ylabel("下视角/°", fontproperties=my_font)
        plt.ylim([20, 45])
        plt.savefig("../../fig/low_orbit_design/zebra_diagram.png", dpi=300)

    def aasr(self, prf, Naz=1):
        len_prf = len(prf)
        Na_band = 1000
        fa_band = np.linspace(-self.Bfov/2, self.Bfov/2, Na_band)
        aasr_num = np.zeros(len_prf)
        aasr = np.zeros(len_prf)
        aasr_deno = 0
        center_apeture = (Naz-1)/2
        if Naz%2 == 0:
            center_apeture = Naz/2
        # 计算方位向响应
        for j in range(len_prf):
            alias_fa = (fa_band) - np.floor((fa_band)/prf[j])*prf[j]
            # 计算重构滤波器
            H_matrix = np.zeros((len(fa_band), Naz, Naz), dtype=complex)
            P_matrix = np.zeros((len(fa_band), Naz, Naz), dtype=complex)
            for k in range(Naz):
                for n in range(Naz):
                    H_matrix[:, k, n] = np.exp(- 1j * np.pi * (n * self.La) / self.Vr * (alias_fa + (k-center_apeture) * prf[j]))


            for k in range(len(fa_band)): 
                P_matrix[k, :, :] = np.linalg.inv(H_matrix[k])

            for i in range(0, 10): 
                ## 天线方向图，这里使用的是矩形天线。如果为其他天线结构，需要更改增益函数
                G_tmp_tx = np.sinc((fa_band + i * prf[j]) * self.La / (2 * self.Vs))**2
                G_tmp_rx = np.sinc((fa_band + i * prf[j]) * self.La / (2 * self.Vs))**2
                G_tmp = G_tmp_rx * G_tmp_tx

                ## 计算方位向重构系统的响应
                H = np.zeros(Naz, dtype=complex)
                pm = np.zeros((len(fa_band)), dtype=complex)
                for t in range(len(fa_band)):
                    for n in range(Naz):
                        H[n] = np.exp(- 1j * np.pi * (n * self.La) / self.Vr * (fa_band[t] + i * prf[j]))    
                    Pa = np.squeeze(P_matrix[t, :, :])
                    Ha = np.squeeze(H)
                    tmp = 0
                    if Naz == 1:
                        pm[t] += Ha*Pa
                    else:
                        band_index = int(np.floor((fa_band[t] + prf[j]*center_apeture)/prf[j]))%Naz
                        tmp = Ha@Pa
                        # if i == 1:
                        #     print(band_index, np.abs(tmp)/np.max(np.abs(tmp)))
                        pm[t] += tmp[band_index]
                                

                G_tmp = G_tmp * np.abs(pm)**2
                if i == 0:
                    aasr_deno = np.trapezoid(G_tmp, fa_band)
                else:
                    aasr_num[j] += 2*np.trapezoid(G_tmp, fa_band) ## +-m

            aasr[j] = aasr_num[j] / aasr_deno 
            aasr[j] = 10 * np.log10(aasr[j])
            
        return aasr

    def nesz(self, doa, Pu, beta):

        cons = 128*np.pi**3 * self.K*self.T * self.Ln / (Pu*self.lambda_**2*self.c)
        R0 = self.calculate_R0(doa)

        ### 天线增益
    
        # A_e = self.d*(self.d) * 0.6 ## 天线有效面积
        # unit_gain = 4*np.pi*A_e/(doa_lambda**2)
        # ant_gain = unit_gain*self.N 
        # ant_gain = 10**(4.5)
        Ae = 0.85*self.La*self.Lr

        ant_gain = 4*np.pi*Ae/self.lambda_**2

        ant_gain = np.sinc(self.Lr*np.sin(doa-beta)/self.lambda_)**2 * ant_gain
        

        # irw = np.deg2rad(2.5)
        # doa35 = np.linspace(-irw/2, (irw/2), len(doa))+beta
        # prm_tmp = self.lambda_
        # prm = np.sin(self.N*(np.pi*(self.ttd*self.c-self.d*np.sin(doa35-beta)))/prm_tmp) / np.sin(np.pi*(self.ttd*self.c-self.d*np.sin(doa35-beta))/prm_tmp)
        # G_r = np.sinc(0.018*np.sin(doa35-beta)/prm_tmp)
        # prm = G_r*prm
        # prm = np.abs(prm)/np.max(np.abs(prm))
        # el_loss = irw/np.trapezoid(prm**4, doa35)
        # self.log.info("el pel loss: {}".format(el_loss))

        # irw_az = np.deg2rad(10.9)
        irw_az = self.theta_a
        # doa35 = np.linspace(-irw_az/2, (irw_az/2), len(doa))+beta
        # G_az = np.sinc(0.04*np.sin(doa35-beta)/self.lambda_)
        # G_az = np.abs(G_az)/np.max(np.abs(G_az))
        # az_loss = irw_az/np.trapezoid(G_az**4, doa35)
        # self.log.info("az pel loss: {}".format(az_loss))


        ## 单一单元增益
        # Ar = 0.6*self.Lr*self.La
        # Gr = (4*np.pi*Ar/(self.lambda_**2))*np.sinc(self.d*np.sin(doa-beta)/self.lambda_)
        # At = 0.6*self.La*self.Lr
        # Gt = (4*np.pi*At/(self.lambda_**2))*np.sinc(self.d*np.sin(doa-beta)/self.lambda_)

        R_eta = R0/np.cos(self.theta_c)
        incident = self.calculate_incident(R_eta)

        var = R0**3 * self.Br * np.sin(incident)/(self.Tp*ant_gain**2 *irw_az)
        nesz = cons*var ## el loss and az loss
        nesz = 10*np.log10(nesz)

        return nesz
    
    def rasr(self, doa, beta, PRF):
        R0 = self.calculate_R0(doa)
        rasr_num = np.zeros(len(doa))
        rasr_dnum = np.zeros(len(doa))
        max_R = np.sqrt((self.H+self.Re)**2 - self.Re**2)

        for m in range(-5, 4):
            Rm = R0 + m*self.c*(1/PRF)/2
            Rm = Rm*(Rm>self.H)*(Rm<max_R)
            if np.any(Rm > 0):
                start_index = np.argmax(Rm > 0)  # 获取Rm不为0的起始点
                end_index = len(Rm) - np.argmax(Rm[::-1] > 0) - 1  # 获取Rm不为0的终止点
            else:
                continue

            window_Rm = slice(start_index, end_index + 1)  # 创建切片对象

            Rm = Rm[window_Rm]
            doam = self.calculate_doa(Rm)


            ## 单一单元增益
            G_doamr = np.sinc(self.Lr*np.sin(doam-beta)/self.lambda_)**2 ## 双程天线增益
            G_doamt = np.sinc(self.Lr*np.sin(doam-beta)/self.lambda_)**2
   
            R_eta = Rm/np.cos(self.theta_c)
            incident = self.calculate_incident(R_eta)
            gain = np.zeros(len(doa))
            gain[window_Rm] = np.squeeze(G_doamr*G_doamt/(Rm**3*np.sin(incident)))
            if m != 0:
                rasr_num += gain
            else:
                rasr_dnum += gain
        rasr = rasr_num/rasr_dnum
        rasr = 10*np.log10(rasr)


        return rasr



        
if __name__ == "__main__":
    design = SlideSpotDesign()
    doa = np.linspace(design.look_angle_left, design.look_angle_right, 1000)
    Pu = 3000

    prf = np.linspace(6.5e3, 9e3, 1000)
    design.zebra_diagram(prf, design.Tp/50)
    nesz = np.array([])
    look_angle = np.array([])
    for i in range(len(design.PRF)):
        doa = np.linspace(design.look_angle_left[i], design.look_angle_right[i], 100)
        nesz_doa = design.nesz(doa, Pu, design.beta[i])
        nesz = np.concatenate([nesz, nesz_doa])
        look_angle = np.concatenate([look_angle, doa])

    print(np.max(nesz))

    plt.figure("NESZ")  
    plt.plot(np.rad2deg(look_angle), nesz, label="NESZ", linewidth=1)
    plt.xlabel("look angle/°")
    plt.ylabel("NESZ/dB", fontproperties=my_font)
    plt.ylim([-28, -15])
    plt.grid()
    plt.legend()
    plt.savefig("../../fig/low_orbit_design/nesz.png", dpi=300)
    # doa = np.linspace(design.look_angle_left, design.look_angle_right, 1000)
    # nesz = design.nesz(doa, Pav)
    rasr = np.array([])
    look_angle = np.array([])

    for i in range(len(design.PRF)):
        doa = np.linspace(design.look_angle_left[i], design.look_angle_right[i], 100)
        rasr_doa = design.rasr(doa, design.beta[i], design.PRF[i])
        rasr = np.concatenate([rasr, rasr_doa])
        look_angle = np.concatenate([look_angle, doa])
    plt.figure("rasr")  
    plt.plot(np.rad2deg(look_angle), rasr, label="RASR", linewidth=1)
    plt.xlabel("look angle/°")
    plt.ylabel("RASR/dB", fontproperties=my_font)
    plt.grid()
    plt.legend()
    plt.savefig("../../fig/low_orbit_design/rasr.png", dpi=300)

    aasr = np.array([])
    for i in range(len(design.PRF)):
        aasr_prf = design.aasr(np.array([design.PRF[i]]), Naz=1)
        aasr = np.concatenate([aasr, aasr_prf])

    plt.figure("AASR")  
    plt.scatter(np.rad2deg(design.beta), aasr, label="AASR")
    plt.plot(np.rad2deg(design.beta), aasr, label="AASR", linewidth=1)
    plt.xlabel("look angle/°")
    plt.ylabel("AASR/dB", fontproperties=my_font)
    plt.grid()
    plt.legend()
    plt.savefig("../../fig/low_orbit_design/aasr.png", dpi=300)

    print(np.mean(design.Tp/ (1/design.PRF))*100)
