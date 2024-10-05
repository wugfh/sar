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
data = zeros(Na+Na/2, Nr+Nr/2);
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

D = sqrt(1-c^2*Ext_f_eta.^2/(4*Vr^2*f0^2));%徙动因子
D_ref = sqrt(1-c^2*fc.^2/(4*Vr^2*f0^2)); % 参考频率处的徙动因子（方位向频率中心）
%大斜视角下，距离调频率随距离变化
K_factor = c*R0*Ext_f_eta.^2./(2*Vr^2*f0^3.*D.^3);
Km = Kr./(1-Kr*K_factor); 

data_ftau_feta = fft(data_tau_feta, Nr, 2);
H_rfm = exp(-4j*pi*(R0-R_ref)/c*sqrt((f0+Ext_f_tau).^2-c^2*Ext_f_eta.^2/(4*Vr^2)));
data_ftau_feta = data_ftau_feta.*H_rfm; %一致rcmc

%% stolt 插值
Ext_f_tau_new = sqrt((f0+Ext_f_tau).^2-c^2*Ext_f_eta.^2/(4*Vr^2))-f0;
max_f_stolt = max(max(Ext_f_tau_new));
min_f_stolt = min(min(Ext_f_tau_new));
f_stolt = fftshift((0:Nr-1)*(max_f_stolt-min_f_stolt)/Nr)+min_f_stolt;

[Ext_f_stolt, Ext_f_eta] = meshgrid(f_stolt, f_eta);
Ext_map_f_tau = sqrt((f0+Ext_f_stolt).^2+c^2*Ext_f_eta.^2/(4*Vr^2))-f0;

sinc_N = 8;
data_ftau_feta_stolt = zeros(Na,Nr);
for i = 1:Na
    for j = 1:Nr
        predict_value = zeros(1, sinc_N);
        map_f_tau = Ext_map_f_tau(i,j);
        map_f_pos = map_f_tau/(Fs/Nr);
        if(map_f_pos<=0)
            map_f_pos = map_f_pos+Nr;
        end
        map_f_int = floor(map_f_pos);
        map_f_remain = map_f_pos-map_f_int;
        sinc_x = map_f_remain - (-sinc_N/2:sinc_N/2-1);
        sinc_y = sinc(sinc_x);
        for m = 1:sinc_N
            if(map_f_pos+m-sinc_N/2 > Nr)
                predict_value(m) = data_ftau_feta(i,Nr);
            elseif(map_f_pos+m-sinc_N/2 < 1)
                predict_value(m) = data_ftau_feta(i,1);
            else
                predict_value(m) = data_ftau_feta(i,map_f_int+m-sinc_N/2);
            end
        end
        data_ftau_feta_stolt(i,j) = sum(predict_value.*sinc_y);
    end
end

%% 成像
Hr = exp(1j*pi*(D./(Km.*D_ref)).*Ext_f_stolt.^2); 
data_ftau_feta_stolt = data_ftau_feta_stolt.*Hr;

data_tau_feta = ifft(data_ftau_feta_stolt, Nr, 2);
R0_RCMC = c*Ext_time_tau_r/2;
Ha = exp(4j*pi*D.*R0_RCMC*f0/c); 
data_tau_feta = data_tau_feta.*Ha; % 方位压缩
data_final = ifft(data_tau_feta, Na, 1);

data_final = abs(data_final)/max(max(abs(data_final)));
data_final = 20*log10(data_final+1);
data_final = data_final.^0.4;
data_final = abs(data_final)/max(max(abs(data_final)));

figure("name","成像结果");
imagesc(abs(data_final));
axis xy;
colormap(gray);