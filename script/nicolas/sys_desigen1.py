import cupy
import numpy 
import time
import matplotlib.pyplot as plt

wave_length = 0.031

EarthMass = 6e24; #地球质量(kg)
EarthRadius = 6.37e6; #地球半径6371km
Gravitational = 6.67e-11; #万有引力常量
PRF = cupy.linspace(1100, 1600, 500)
incident_min = cupy.deg2rad(20)
incident_max = cupy.deg2rad(55)
H = 1e3*cupy.linspace(500, 700, 500)

squint_min = cupy.arcsin(EarthRadius*cupy.sin(incident_min)/(H+EarthRadius))
beta_min = incident_min - squint_min

squint_max = cupy.arcsin(EarthRadius*cupy.sin(incident_max)/(H+EarthRadius))
beta_max = incident_max - squint_max

Wg = EarthRadius*(beta_max - beta_min)

plt.plot(cupy.asnumpy(H), cupy.asnumpy(Wg))
# plt.show()

H = 580e3 #卫星高度
Vr = cupy.sqrt(Gravitational*EarthMass/(EarthRadius + H)); #第一宇宙速度
Vg = Vr*EarthRadius/(EarthRadius + H); #地面速度

Tr = 40e-6  #脉冲重复周期

