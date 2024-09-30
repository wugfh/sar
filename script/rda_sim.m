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
data = zeros(Na+Na, Nr+Nr/2);
data(1:Na, 1:Nr) = data_1;
[Na,Nr] = size(data);
kai = kaiser(Nr, 3);
Ext_kai = repmat(kai', Na, 1);
data = data.*Ext_kai;

R0 = start*c/2;
theta_rc = asin(fc*lambda/(2*Vr));
Ka = 2*Vr^2*cos(theta_rc)^3/(lambda*R0);
R_eta_c = R0/cos(theta_rc);
eta_c = 2*Vr*sin(theta_rc)/lambda;

f_tau = fftshift((-Nr/2:Nr/2-1)*(Fs/Nr));
f_eta = fc + (-Na/2:Na/2-1)*(Fa/Na);

tau = 2*R_eta_c/c + (-Nr/2:Nr/2-1)*(1/Fs);
eta = eta_c + (-Na/2:Na/2-1)*(1/Fa);

[Ext_time_tau_r, Ext_time_eta_a] = meshgrid(tau, eta);
[Ext_f_tau, Ext_f_eta] = meshgrid(f_tau, f_eta);


%% 小斜视角

% 距离压缩
data_fft_r = fft(data, Nr, 2);
Hr = (abs(Ext_f_tau) < B/2).*exp(1j*pi*Ext_f_tau.^2/Kr);
data_fft_cr = data_fft_r.*Hr;
data_cr = ifft(data_fft_cr, Nr, 2);

% 距离徙动校正
data_fft_a = fft(data_cr, Na, 1);

data_fft_a_rcmc = data_fft_a;
IN_N = 8;
R0_RCMC = tau*c/2;  
delta_R = lambda^2*f_eta'.^2.*R0_RCMC/(8*Vr^2);
delta_R_cnt = delta_R*2/(c*(1/Fs));
for j = 1:Na
    for k = 1:Nr
        dR = delta_R_cnt(j,k);
        pos = dR-floor(dR)-(-IN_N/2:IN_N/2-1);
        rcmc_sinc = sinc(pos);
        size_sinc = size(rcmc_sinc);
        predict_value = zeros(size_sinc);
        for m = -IN_N/2:IN_N/2-1
            if(k+floor(dR)+m>Nr)
                predict_value(m+IN_N/2+1) = data_fft_a(j,k+floor(dR)+m-Nr);
            else
                predict_value(m+IN_N/2+1) = data_fft_a(j,k+floor(dR)+m);
            end
        end
        data_fft_a_rcmc(j,k) = sum(predict_value.*rcmc_sinc);
    end
end

% 方位压缩
Ha = exp(-1j*pi*Ext_f_eta.^2./Ka);
offset = exp(1j*2*pi*Ext_f_eta.*eta_c);
% offset2 = exp(1j*2*pi*Ext_f_eta.*Ext_time_tau_r*2
% offset = 1;
offset2 = 1;
data_fft_ca_rcmc = data_fft_a_rcmc.*Ha.*offset.*offset2;
data_ca_rcmc = ifft(data_fft_ca_rcmc, Na, 1);

data_final = data_ca_rcmc;

% data_final(:,1:Nr-Nr_tmp+1) = data_ca_rcmc(:,Nr_tmp:Nr);
% data_final(:,Nr-Nr_tmp+1+1:Nr) = data_ca_rcmc(:,1:Nr_tmp-1);

data_tmp = data_final;

data_final(1:Na-Na_tmp+1,:) = data_tmp(Na_tmp:Na,:);
data_final(Na-Na_tmp+1+1:Na, :) = data_tmp(1:Na_tmp-1,:);


data_final = abs(data_final)/max(max(abs(data_final)));
data_final = log10(abs(data_final)+1);
data_final = data_final.^0.4;
data_final = abs(data_final)/max(max(abs(data_final)));
figure;
imagesc(data_final);
axis xy;
colormap(gray);