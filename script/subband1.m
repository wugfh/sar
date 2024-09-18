clear; clc;

f0 = 9.6e9; % 中心频率
vs = 124; % 传感器速度
H = 7052; % 飞行高度
Tr = 20e-6; % 脉冲宽度
sub_f1 = 8.4e9; % 子带1中心频率
sub_f2 = 9.6e9; % 子带2中心频率
sub_f3 = 10.8e9; % 子带3中心频率
Br = 1.2e9; % 子带带宽
Fr = 1.4e9; % 子带采样率
PRF = 2000; 
c = 299792458;

Kr = B/Tr;

phi = deg2rad(20);
incidence = deg2rad(20.5);

R0 = H/cos(phi);
R_eta_c = H/cos(incidence);

Nr = 2048;
Na = 2048;

t_tau = 2*R_eta_c/c;