import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import fsolve
from scipy.signal import decimate
from multiprocessing import Process, Queue

from matplotlib import font_manager

my_font = font_manager.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


class SlideSpotDesign:
    def __init__(self):
        self.H = 200e3
        self.c = 299792458 # Speed of light in m/s
        self.EarthMass = 5.972e24 # kg
        self.Re = 6371e3
        self.Gravitational = 6.67430e-11
        self.Vs = np.sqrt(self.Gravitational*self.EarthMass/(self.Re + self.H))  
        self.Vg = self.Vs * self.Re / (self.Re + self.H)
        self.da = 0.1
        self.dr = 0.1
        self.Bd = 0.886*self.Vg/self.da
        self.f0 = 35e9
        self.Tp = 20e-6
        self.lambda_ = self.c / self.f0
        self.beta = np.deg2rad(20) ## 下视角中心
        tmp_angle = np.arcsin((self.H+self.Re)*np.sin(self.beta)/self.Re)
        tmp_angle = tmp_angle - self.beta
        self.R0 = self.Re*np.sin(tmp_angle)/np.sin(self.beta)
        self.PRF = 6000
        self.NB = 1

        self.psi_start = np.deg2rad(60)
        self.psi_end = np.deg2rad(65)       

        def Bfov_func(theta_a):
            return 2*self.Vs*np.abs(np.sin(self.psi_start+theta_a/2) - np.sin(self.psi_start-theta_a/2))/self.lambda_-self.PRF+500

        self.theta_a = fsolve(Bfov_func, np.deg2rad(0.1))[0]     

        self.Bd = 2*self.Vs*np.abs(np.tan(self.psi_start)/np.sqrt(1+np.tan(self.psi_start)**2) 
                             - np.tan(self.psi_end)/np.sqrt(1+np.tan(self.psi_end)**2)) / (self.lambda_)
        
    
    def calulate_extent(self, psi_start, psi_end, theta_a, R0):
        dstart = R0 * np.tan(psi_start)
        dend = R0 * np.tan(psi_end)
        dstart_1 = R0*(np.tan(psi_start-theta_a/2)) - dstart
        dstart_2 = R0*(np.tan(psi_start+theta_a/2)) - dstart
        dend_1 = R0*(np.tan(psi_end-theta_a/2)) - dend
        dend_2 = R0*(np.tan(psi_end+theta_a/2)) - dend
        if dend_2 < dstart_1 or dend_1 > dstart_2:
            print("Error: The extent of the beam does not overlap.")
            return 0
        return min(dstart_2, dend_2) - max(dstart_1, dend_1)
            
    
    def zebra_diagram(self, prf, tau_rp):

        frac_min = -(tau_rp + self.Tp) * prf  # 发射约束
        frac_max = tau_rp * prf

        int_min = int(np.min(np.floor(2 * self.H * prf / self.c)))
        int_max = int(np.max(np.ceil(2 * np.sqrt((self.Re+self.H)**2 - (self.Re)**2) * prf / self.c)))

        plt.figure("PRF selection")

        # 遍历尽可能多的干扰
        for i in range(int_max + 1):
            R1 = (i + frac_min) * self.c / (2 * prf)
            gamma_cos = (R1**2 + (self.Re+self.H)**2 - self.Re**2) / (2 * R1 * (self.Re+self.H))
            gamma_cos = np.clip(gamma_cos, -1, 1)
            gamma = np.arccos(np.abs(gamma_cos))
            # belta = np.arcsin(R1 * np.sin(gamma) / self.Re)
            gamma1 = np.rad2deg(gamma)

            plt.plot(prf, gamma1, 'b')

            Rn = (i + frac_max) * self.c / (2 * prf)
            gamma_cos = (Rn**2 + (self.Re+self.H)**2 - self.Re**2) / (2 * Rn * (self.Re+self.H))
            gamma_cos = np.clip(gamma_cos, -1, 1)
            gamma = np.arccos(np.abs(gamma_cos))
            # belta = np.arcsin(Rn * np.sin(gamma) / self.Re)
            gamma2 = np.rad2deg(gamma)

            plt.plot(prf, gamma2, 'b')
            plt.fill_between(prf, gamma1, gamma2, alpha=0.1, color='b')

        # 星下点干扰
        for i in range(6):
            R = (2 * self.H / self.c + i / prf) * self.c / 2
            gamma_cos = (R**2 + (self.Re+self.H)**2 - self.Re**2) / (2 * R * (self.Re+self.H))
            gamma_cos = np.clip(gamma_cos, -1, 1)
            gamma = np.arccos(np.abs(gamma_cos))
            # belta = np.arcsin(R * np.sin(gamma) / self.Re)
            gamma1 = np.rad2deg(gamma)

            plt.plot(prf, gamma1, 'r')

            R = (2 * self.H / self.c + i / prf + self.Tp * 2) * self.c / 2
            gamma_cos = (R**2 + (self.Re+self.H)**2 - self.Re**2) / (2 * R * (self.Re+self.H))
            gamma_cos = np.clip(gamma_cos, -1, 1)
            gamma = np.arccos(np.abs(gamma_cos))
            # belta = np.arcsin(R * np.sin(gamma) / self.Re)
            gamma2 = np.rad2deg(gamma)

            plt.plot(prf, gamma2, 'r')
            plt.fill_between(prf, gamma1, gamma2, alpha=0.1, color='r')
        
    
        plt.grid()
        plt.xlabel("PRF/Hz")
        plt.ylabel("下视角/°", fontproperties=my_font)
        plt.ylim([15, 50])
        plt.savefig("../../fig/low_orbit_design/zebra_diagram.png", dpi=300)

        

        
if __name__ == "__main__":
    design = SlideSpotDesign()
    print(0.886*design.Vg/design.da)
    print(np.rad2deg(design.theta_a))
    print(design.Bd)
    print(design.calulate_extent(design.psi_start, design.psi_end, design.theta_a, design.R0))
    design.zebra_diagram(np.linspace(2e3, 10e3, 1000), design.Tp/10)