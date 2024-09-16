clear,clc;

load("../data/RadarSAT数据/RadarSAT数据/data_1.mat");

c = 299792458;                     %光速
Fs = 32317000;      %采样率                                   
start = 6.5959e-03;         %开窗时间 
Tr = 4.175000000000000e-05;        %脉冲宽度                        
f0 = 5.300000000000000e+09;                    %载频                     
PRF = 1.256980000000000e+03;       %PRF                     
Vr = 7062;                       %雷达速度     
B = 30.111e+06;        %信号带宽
fc = -6900;          %多普勒中心频率
Ka = 1733;
Fa = PRF;

lambda = c/f0;
Kr = -B/Tr;
[Na, Nr] = size(data_1);
theta_rc = asin(fc*lambda/(2*Vr));
R_eta_c = 2*Vr^2*cos(theta_rc)^2/(lambda*Ka);
R0 = R_eta_c*cos(theta_rc);
eta_c = 2*Vr*sin(theta_rc)/lambda;

f_tau = (-Nr/2:Nr/2-1)*(Fs/Nr);
f_eta = fc + (-Na/2:Na/2-1)*(Fa/Na);

tau = 2*R_eta_c/c + (-Nr/2:Nr/2-1)*(1/Fs);
eta = eta_c + (-Na/2:Na/2-1)*(1/Fa);

[Ext_time_tau_r, Ext_time_eta_a] = meshgrid(tau, eta);
[Ext_f_tau, Ext_f_eta] = meshgrid(f_tau, f_eta);

%% 小斜视角

% 距离压缩
data_fft_r = fftshift(fft(fftshift(data_1, 2), Nr, 2), 2);
Hr = (abs(Ext_f_tau) < B/2).*exp(1j*pi*Ext_f_tau.^2/Kr);
data_fft_cr = data_fft_r.*Hr;
data_cr = fftshift(ifft(fftshift(data_fft_cr, 2), Nr, 2), 2);

% 距离徙动校正
data_fft_a = fftshift(fft(fftshift(data_cr, 1), Na, 1), 1);

data_fft_a_rcmc = data_fft_a;
IN_N = 8;
R0_RCMC = tau*c/2;  % 每个距离门都是一个R0
delta_R = lambda^2*f_eta'.^2.*R0_RCMC/(8*Vr^2);
delta_R_cnt = delta_R*2/(c*(1/Fs));
for j = 1:Na
    for k = 1:Nr
        data_fft_a_rcmc(j,k) = 0;
        dR = delta_R_cnt(j,k);
        for m = -IN_N/2:IN_N/2-1
            if(k+floor(dR)+m>=Nr)
                data_fft_a_rcmc(j,k) = data_fft_a_rcmc(j,k)+data_fft_a(j,Nr)*sinc(dR-(Nr-k));
            elseif(k+floor(dR)+m<=1)
                data_fft_a_rcmc(j,k) = data_fft_a_rcmc(j,k)+data_fft_a(j,1)*sinc(dR-(1-k));
            else
                data_fft_a_rcmc(j,k) = data_fft_a_rcmc(j,k)+data_fft_a(j,k+floor(dR)+m)*sinc(dR-floor(dR)-m);
            end
        end
    end
end

% 方位压缩
Ha = exp(-1j*pi*Ext_f_eta.^2/Ka);
Offset = exp(-1j*2*pi*Ext_f_eta.*eta_c);
data_fft_ca_rcmc = data_fft_a_rcmc.*Ha.*Offset;
data_ca_rcmc = fftshift(ifft(fftshift(data_fft_ca_rcmc, 1), Na, 1), 1);

data_fft_ca = data_fft_a.*Ha.*Offset;
data_ca = fftshift(ifft(fftshift(data_fft_ca, 1), Na, 1), 1);

data_ca = 20*log(abs(data_ca)+1);
data_ca_rcmc = 20*log(abs(data_ca_rcmc)+1);

figure;
subplot(121)
imagesc(abs(data_1));
title("原始数据");
subplot(122)
imagesc(abs(data_ca_rcmc));
title("处理后的数据");

figure;
subplot(121)
imagesc(data_ca);
title("无rcmc");
subplot(122)
imagesc(data_ca_rcmc);
title("有rcmc");

