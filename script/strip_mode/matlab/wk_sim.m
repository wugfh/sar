clear,clc;

tic;
%% 参数与初始化
load("../../../data/English_Bay_ships/data_1.mat");

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
data(Na/2:Na/2+Na-1, Nr/2:Nr/2+Nr-1) = data_1;
[Na,Nr] = size(data);


R0 = start*c/2;
theta_rc = asin(fc*lambda/(2*Vr));
Ka = 2*Vr^2*cos(theta_rc)^3/(lambda*R0);
R_eta_c = R0/cos(theta_rc);
eta_c = 2*Vr*sin(theta_rc)/lambda;

f_tau = fftshift((-Nr/2:Nr/2-1)*(Fs/Nr));
f_eta = fc + (((-Na/2:Na/2-1)*(Fa/Na)));

tau = 2*R_eta_c/c + (-Nr/2:Nr/2-1)*(1/Fs);
eta = eta_c + (-Na/2:Na/2-1)*(1/Fa);

[Ext_time_tau_r, Ext_time_eta_a] = meshgrid(tau, eta);
[Ext_f_tau, Ext_f_eta] = meshgrid(f_tau, f_eta);

R_ref = R_eta_c; 

%% 一致RCMC
data_ftau_feta = fft2(data); 

H_rfm = exp(4j*pi*(R_ref)/c*sqrt((f0+Ext_f_tau).^2-c^2*Ext_f_eta.^2/(4*Vr^2)) + 1j*pi*Ext_f_tau.^2/Kr);
data_ftau_feta = data_ftau_feta.*H_rfm; %一致rcmc

Ext_map_f_tau = sqrt((f0+Ext_f_tau).^2+c^2*Ext_f_eta.^2/(4*Vr^2))-f0; %线性变化的stolt频率轴与原始频率轴的对应（stolt 映射）

delta = (Ext_map_f_tau - Ext_f_tau)/(Fs/Nr);
delta_int = floor(delta);
delta_remain = delta-delta_int;

%插值使用8位 sinc插值
sinc_N = 8;
data_ftau_feta_stolt = zeros(Na,Nr);
for i = 1:Na
    for j = 1:Nr
        predict_value = zeros(1, sinc_N);
        dR_int = delta_int(i,j);
        sinc_x = delta_remain(i,j) - (-sinc_N/2:sinc_N/2-1);
        sinc_y = sinc(sinc_x);
        for m = 1:sinc_N
            index = dR_int+m+j-sinc_N/2;
            if(index > Nr)
                predict_value(m) = data_ftau_feta(i,Nr);
            elseif(index < 1)
                predict_value(m) = data_ftau_feta(i,1);
            else
                predict_value(m) = data_ftau_feta(i,index);
            end
        end
        data_ftau_feta_stolt(i,j) = sum(predict_value.*sinc_y)/sum(sinc_y);
    end
end

%% 成像
% data_ftau_feta_stolt = data_ftau_feta;
% data_ftau_feta_stolt = data_ftau_feta_stolt.*exp(-4j*pi*R_ref*Ext_f_tau/c);
data_final = fftshift(ifft2(data_ftau_feta_stolt), 1);
data_final = flip(data_final, 1);

%简单的后期处理
data_final = abs(data_final)/max(max(abs(data_final)));
data_final = 20*log10(data_final+1);
data_final = data_final.^0.4;
data_final = abs(data_final)/max(max(abs(data_final)));

figure("name","成像结果");
imshow(data_final)


data_final = fftshift(ifft2(data_ftau_feta), 1);
data_final = flip(data_final, 1);
data_final = abs(data_final)/max(max(abs(data_final)));
data_final = 20*log10(data_final+1);
data_final = data_final.^0.4;
data_final = abs(data_final)/max(max(abs(data_final)));
figure("name", "no stolt");
imshow(data_final)
title("no stolt")

toc;
fprintf("运行时间：%f s\n", toc);