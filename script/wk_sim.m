clear,clc;

load("../data/English_Bay_ships/data_1.mat");

c = 299792458;                     %光速
Fs = 32317000;      %采样率                                   
start = 6.5959e-03;         %开窗时间 
Tr = 4.175000000000000e-05;        %脉冲宽度                        
f0 = 5.300000000000000e+09;                    %载频                     
PRF = 1.256980000000000e+03;       %PRF                     
Vr = 7062;                       %雷达速度     
B = 30.111e+06;        %信号带宽
fc = -6900;          %多普勒中心频率
Fa = PRF;

% Ka = 1733;

lambda = c/f0;
Kr = -B/Tr;
% Kr = -7.2135e+11;

[Na_tmp, Nr_tmp] = size(data_1);
kai = kaiser(Nr_tmp, 2.5);
Ext_kai = repmat(kai', Na_tmp, 1);
data_1 = data_1.*Ext_kai;
[Na, Nr] = size(data_1);
data = zeros(Na+Na, Nr+Nr);
data(Na/2:Na+Na/2-1, Nr/2:Nr/2+Nr-1) = data_1;
[Na,Nr] = size(data);


R0 = start*c/2;
theta_rc = asin(fc*lambda/(2*Vr));
Ka = 2*Vr^2*cos(theta_rc)^3/(lambda*R0);
R_eta_c = R0/cos(theta_rc);
eta_c = 2*Vr*sin(theta_rc)/lambda;

f_tau = fftshift((-Nr/2:Nr/2-1)*(Fs/Nr));
f_eta = fc + fftshift((-Na/2:Na/2-1)*(Fa/Na));

tau = 2*R_eta_c/c + (-Nr/2:Nr/2-1)*(1/Fs);
eta = eta_c + (-Na/2:Na/2-1)*(1/Fa);

[Ext_time_tau_r, Ext_time_eta_a] = meshgrid(tau, eta);
[Ext_f_tau, Ext_f_eta] = meshgrid(f_tau, f_eta);

R_ref = R_eta_c; % 将参考目标设为场景中心


data = data.*exp(-2j*pi*fc*Ext_time_eta_a);
data_tau_feta = fft(data, Na, 1); % 首先变换到距离多普勒域

%% 成像

D = sqrt(1-c^2*Ext_f_eta.^2/(4*Vr^2*f0^2));%徙动因子
D_ref = sqrt(1-c^2*fc.^2/(4*Vr^2*f0^2)); % 参考频率处的徙动因子（方位向频率中心）

rcm_total = R0./D - R0/D_ref; % 整体rcm
rcm_bulk = R_ref./D - R_ref/D_ref; % 一致rcm（参考点的rcm）
rcm_diff = rcm_total - rcm_bulk; % 补余rcm（整体rcm与参考rcm的差）

