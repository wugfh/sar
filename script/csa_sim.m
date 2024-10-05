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
data = zeros(Na, Nr);
data(1:Na, 1:Nr) = data_1;
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

%大斜视角下，距离调频率随距离变化
K_factor = c*R0*Ext_f_eta.^2./(2*Vr^2*f0^3.*D.^3);
Km = Kr./(1-Kr*K_factor); 

delta_tau = 2*rcm_diff/c; % 补余rcmc的变标移动的时间
f_sc = Km.*delta_tau; % 变标的频率偏移

Ext_echo_tau = Ext_time_tau_r-2*R_ref./(c*D); %以参考目标为中心

s_sc = exp(2j*pi*Km.*Ext_echo_tau.*delta_tau);
data_tau_feta = data_tau_feta.*s_sc; %变标，完成补余rcmc

data_ftau_feta = fft(data_tau_feta, Nr, 2);
data_ftau_feta = data_ftau_feta.*exp(2j*pi*(2*rcm_bulk/c).*Ext_f_tau); % 一致rcmc

Hr = exp(1j*pi*(D./(Km.*D_ref)).*Ext_f_tau.^2); 
data_ftau_feta = data_ftau_feta.*Hr; % 距离压缩

data_tau_feta = ifft(data_ftau_feta, Nr, 2);

R0_RCMC = c*Ext_time_tau_r/2;
Ha = exp(4j*pi*D.*R0_RCMC*f0/c); 
data_tau_feta = data_tau_feta.*Ha; % 方位压缩
offset = exp(-4j*pi*Km.*(1-D./D_ref).*(R0./D-R_ref./D).^2/c^2);
data_tau_feta = data_tau_feta.*offset; %附加相位校正;

data_final = ifft(data_tau_feta, Na, 1);

data_final = abs(data_final)/max(max(abs(data_final)));
data_final = 20*log10(abs(data_final)+1);
data_final = data_final.^0.4;
data_final = abs(data_final)/max(max(abs(data_final)));

data_final = fftshift(data_final, 1);

figure("name","成像结果");
imagesc(abs(data_final));
axis xy;
colormap(gray);