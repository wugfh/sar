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
% R0 = 9.886474620000000e+05;
Ka = 1733;

lambda = c/f0;
Kr = -B/Tr;

data = data_1;
% [Na, Nr] = size(data_1);
% data = zeros(Na, Nr+500);
% data(1:Na, 1:Nr) = data_1;
[Na, Nr] = size(data);


theta_rc = asin(fc*lambda/(2*Vr));
R0 = 2*Vr^2*cos(theta_rc)^3/(lambda*Ka);
R_eta_c = R0/cos(theta_rc);
eta_c = 2*Vr*sin(theta_rc)/lambda;

f_tau = (-Nr/2:Nr/2-1)*(Fs/Nr);
f_eta = fc + (-Na/2:Na/2-1)*(Fa/Na);

tau = 2*R_eta_c/c + (-Nr/2:Nr/2-1)*(1/Fs);
eta = eta_c + (-Na/2:Na/2-1)*(1/Fa);


[Ext_time_tau_r, Ext_time_eta_a] = meshgrid(tau, eta);
[Ext_f_tau, Ext_f_eta] = meshgrid(f_tau, f_eta);

%% 大斜视角

% 距离压缩
D = sqrt(1-lambda^2*f_eta.^2/(4*Vr^2)); % 徙动因子
Ext_D = repmat(D', 1, Nr);

Ksrc = 2*Vr^2*f0^3*Ext_D.^3./(c*R0*Ext_f_eta.^2);


data_fft_r = fft(data, Nr, 2);
Hr = (abs(Ext_f_tau) < B/2).*exp(1j*pi*Ext_f_tau.^2/Kr);
Hm = (abs(Ext_f_tau) < B/2).*exp(-1j*pi*Ext_f_tau.^2./Ksrc);
data_fft_cr = data_fft_r.*Hr.*Hm;
data_cr = ifft(data_fft_cr, Nr, 2);


% 距离徙动校正
data_fft_a = fft(data_cr, Na, 1);

data_fft_a_rcmc = data_fft_a;
IN_N = 8;
R0_RCMC = tau*c/2;  

for j = 1:Na
    for k = 1:Nr
        data_fft_a_rcmc(j,k) = 0;
        dR = R0_RCMC(k)/(D(j)) - R0_RCMC(k);
        dR = dR*2/(c*(1/Fs));
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
Ha = exp(4j*pi*D'*R0_RCMC*f0./c);
offset = exp(1j*2*pi*Ext_f_eta.*eta_c);
data_fft_ca_rcmc = data_fft_a_rcmc.*Ha.*offset;
data_ca_rcmc = ifft(data_fft_ca_rcmc, Na, 1);

data_fft_ca = data_fft_a.*Ha.*offset;
data_ca = ifft(data_fft_ca, Na, 1);

data_ca_rcmc = abs(data_ca_rcmc)/max(max(abs(data_ca_rcmc)));
data_ca = abs(data_ca)/max(max(abs(data_ca)));
data_ca = 20*log(abs(data_ca)+eps);
data_ca_rcmc = 20*log(abs(data_ca_rcmc)+eps);

data_ca = mat2gray(data_ca);
data_ca = histeq(data_ca);
data_ca_rcmc = mat2gray(data_ca_rcmc);
data_ca_rcmc = histeq(data_ca_rcmc);


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
axis xy;
colormap(gray)
title("无rcmc");
subplot(122)
imagesc(data_ca_rcmc);
axis xy;
colormap(gray)
title("有rcmc");

figure;
imagesc(data_ca_rcmc);
axis xy;
colormap(gray)

