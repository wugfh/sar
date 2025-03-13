clear; clc;

%% 参数

% 轨道参数
c = 299792458;
H = 755e3;
phi = 20*pi/180;
incidence = 20.5*pi/180;
% incidence = phi;
R_eta_c = H/cos(incidence);
R0 = H/cos(phi);
theta_rc = acos(R0/R_eta_c);

% 卫星参数
Vr = 7062;
Vs = Vr;
Vg = Vr;
La = 15;
Lr = 1.5;
f0 = 5.4e9;
lambda = c/f0;

% 距离向

Tr = 40e-6;
Br = 2.8*6e6;
Kr = Br/Tr;
alpha_osr = 1.2;
Fr = alpha_osr*Br;
Nr = ceil(1.5*Fr*Tr);

% 方位向

alpha_osa = 1.7;
delta_fdop = 2*0.886*Vs*cos(theta_rc)/La; 
Fa = delta_fdop*alpha_osa;
Ta = 0.886*R_eta_c*lambda/(La*Vg*cos(theta_rc));
Na = ceil(1.2*Fa*Ta);

% 景中心

time_eta_c = -R0*tan(theta_rc)/Vr;
f_eta_c = 2*Vr*sin(theta_rc)/lambda;

% 合成孔径

rho_r = c/(2*Fr);
rh0_a = La/2;
theta_bw = 0.886*lambda/La;
theta_syn = Vs/Vg*theta_bw;
Ls = R_eta_c * theta_syn;

Trg = Nr/Fr;Taz = Na/Fa;
Gap_t_tau = 1/Fr; Gap_t_eta = 1/Fa;
Gap_f_tau = Fr/Nr;Gap_f_eta = Fa/Na;
time_tau_r = 2*R_eta_c/c + (-Trg/2:Gap_t_tau:Trg/2-Gap_t_tau);
time_eta_a = time_eta_c+(-Taz/2:Gap_t_eta:Taz/2-Gap_t_eta);

R0_tau_r = (time_tau_r*c/2)*cos(theta_rc);
Ext_R0_tau_r = repmat(R0_tau_r, Na, 1);

f_tau = (-Fr/2:Gap_f_tau:Fr/2-Gap_f_tau);
f_tau = f_tau-(round(f_tau/Fr))/Fr;
f_eta = f_eta_c+(-Fa/2:Gap_f_eta:Fa/2-Gap_f_eta);
f_eta = f_eta-(round(f_eta/Fa))/Fa;

[Ext_time_tau_r, Ext_time_eta_a] = meshgrid(time_tau_r, time_eta_a);
[Ext_f_tau, Ext_f_eta] = meshgrid(f_tau, f_eta);

%% 点目标
xA = 0; yA = 0;
xB = xA;yB = yA+500;
xC = H*tan(phi+theta_bw/2)-R0*sin(phi);
yC = yA+500;
pos_x = [xC];
pos_y = [yC];

%% 回波产生
target_cnt = 1;
S_echo = zeros(Na, Nr);
for i = 1:target_cnt
    R0_tar =  sqrt((R0 * sin(phi) + pos_x(i))^2+H^2);
    time_eta_c_tar = (pos_y(i) - R0_tar*tan(theta_rc))/Vr;
    R_eta = sqrt(R0_tar^2+(Vr*Ext_time_eta_a-pos_y(i)).^2);
    Wr = (abs(Ext_time_tau_r-2*R_eta/c) <= Tr/2);
    Wa = sinc(La*sin(acos(R0_tar./R_eta)-theta_rc)/lambda).^2;
%     Wa = (La*atan(Vg*(Ext_time_eta_a - time_eta_c_tar)./(R0 * sin(phi) + pos_x(i))/lambda).^2)<=Ta/2;
    Phase = exp(-4j*pi*f0*R_eta/c).*exp(1j*pi*Kr*(Ext_time_tau_r-2*R_eta/c).^2);
    S_echo_tar = Wr.*Wa.*Phase;
    S_echo = S_echo+S_echo_tar;
    R_eta_c_tar = R0_tar/cos(theta_rc);
    fprintf("当前目标:%d,坐标:(%.2f,%.2f),\n最近斜距R0=%.2f,景中心斜距R_eta_c=%.2f,波束中心穿越时刻=%.4f\n", i, pos_x(i), pos_y(i), R0_tar, R_eta_c_tar,time_eta_c_tar)
    time_tau_p = round((2*R0_tar/(c*cos(theta_rc)) - time_tau_r(1))/Gap_t_tau);
    time_eta_p = Na/2+(pos_y(i)/Vr)/Gap_t_eta;
    fprintf("徙动校正前,点数坐标应为%d行(方),%d列(距)\n\n",round(time_eta_p),time_tau_p);
end
S_echo = S_echo .* exp(-2j*pi*f_eta_c*Ext_time_eta_a);
%% 回波图可视化

figure('name', "回波可视化")
subplot(2, 2, 1);
imagesc(real(S_echo));
title('原始仿真信号实部');
xlabel('距离向时间 \tau');
ylabel('方位向时间 \eta');

subplot(2, 2, 2);
imagesc(imag(S_echo));
title('原始仿真信号虚部');
xlabel('距离向时间 \tau');
ylabel('方位向时间 \eta');


subplot(2, 2, 3);
imagesc(abs(S_echo));
title('原始仿真信号幅度');
xlabel('距离向时间 \tau');
ylabel('方位向时间 \eta');

subplot(2, 2, 4);
imagesc(angle(S_echo));
title('原始仿真信号相位');
xlabel('距离向时间 \tau');
ylabel('方位向时间 \eta');

%% 回波频域
S_tau_feta = fftshift(fft(fftshift(S_echo, 1), Na, 1), 1);
S_ftau_feta = fft2(S_echo, Na, Nr);
figure('name', "回波频域")
subplot(2, 2, 1);
imagesc(abs(S_ftau_feta));
title('二维频域幅度');
xlabel('距离向频谱点数 f_\tau');
ylabel('方位向频谱点数 f_\eta');

subplot(2, 2, 2);
imagesc(angle(S_ftau_feta));
title('二维频域相位');
xlabel('距离向频谱点数f_\tau');
ylabel('方位向频谱点数f_\eta');

subplot(2, 2, 3);
imagesc(abs(S_tau_feta));
title('RD域幅度');
xlabel('距离向时间 \tau');
ylabel('方位向频谱点数f_\eta');

subplot(2, 2, 4);
imagesc(angle(S_tau_feta));
title('RD域相位');
xlabel('距离向时间 \tau');
ylabel('方位向频谱点数f_\eta');



%% 距离压缩
Hf = (abs(Ext_f_tau)<=Br/2).*exp(1j*pi*Ext_f_tau.^2/Kr);
S1_ftau_eta = fftshift(fft(fftshift(S_echo, 2), Nr, 2),2);
S1_ftau_eta = S1_ftau_eta.*Hf;
S1_tau_eta = fftshift(ifft(fftshift(S1_ftau_eta,2), Nr, 2),2);
%% 距离压缩可视化

%距离压缩结果
figure('name', "距离压缩时域结果")
subplot(1, 2, 1);
imagesc(real(S1_tau_eta));
title('实部');
xlabel('距离向时间 \tau');
ylabel('方位向时间 \eta');

subplot(1, 2, 2);
imagesc(abs(S1_tau_eta));
title('幅度');
xlabel('距离向时间 \tau');
ylabel('方位向时间 \eta');

%% 方位向fft
S2_tau_feta = fftshift(fft(fftshift(S1_tau_eta, 1), Na, 1), 1);

%% 方位向傅里叶变换可视化

figure('name', "方位向傅里叶变换结果")
subplot(1, 2, 1);
imagesc(real(S2_tau_feta));
title('实部');
xlabel('距离向时间\tau');
ylabel('方位向频谱点数f_\eta');

subplot(1, 2, 2);
imagesc(abs(S2_tau_feta));
title('幅度');
xlabel('距离向时间\tau');
ylabel('方位向频谱点数f_\eta');

%% RCMC, 相位补偿法
% delta_R = lambda^2*R0*Ext_f_eta.^2/(8*Vr^2);
% G_rcmc = exp(1j*4*pi*Ext_f_tau.*delta_R/c);
% S3_ftau_feta = fftshift(fft(fftshift(S2_tau_feta,2), Nr, 2), 2);
% S3_ftau_feta = S3_ftau_feta.*G_rcmc;
% S3_tau_feta_RCMC = fftshift(ifft(fftshift(S3_ftau_feta,2),Nr, 2), 2);

%% RCMC，平移RCM
S3_tau_feta_RCMC = S2_tau_feta;
IN_N = 8;
R0_RCMC = time_tau_r*c/2;  % 每个距离门都是一个R0
R0_tar =  sqrt((R0 * sin(phi) + pos_x(i))^2+H^2);
delta_R = lambda^2*f_eta'.^2.*R0_RCMC/(8*Vr^2);
delta_R_cnt = delta_R*2/(c*Gap_t_tau);
for j = 1:Na
    for k = 1:Nr
        S3_tau_feta_RCMC(j,k) = 0;
        dR = delta_R_cnt(j,k);
        for m = -IN_N/2:IN_N/2-1
            if(k+floor(dR)+m>=Nr)
                S3_tau_feta_RCMC(j,k) = S3_tau_feta_RCMC(j,k)+S2_tau_feta(j,Nr)*sinc(dR-(Nr-k));
            elseif(k+floor(dR)+m<=1)
                S3_tau_feta_RCMC(j,k) = S3_tau_feta_RCMC(j,k)+S2_tau_feta(j,1)*sinc(dR-(1-k));
            else
                S3_tau_feta_RCMC(j,k) = S3_tau_feta_RCMC(j,k)+S2_tau_feta(j,k+floor(dR)+m)*sinc(dR-floor(dR)-m);
            end
        end
    end
end

%% 无RCMC
% S3_tau_feta_RCMC = S2_tau_feta;

%% 距离徙动校正可视化

figure('name', "距离徙动校正结果")
subplot(1, 2, 1);
imagesc(abs(S2_tau_feta));
title('RCMC前');
xlabel('距离向时间\tau');
ylabel('方位向频谱点数 f_\eta');

subplot(1, 2, 2);
imagesc(abs(S3_tau_feta_RCMC));
title('RCMC后');
xlabel('距离向时间\tau');
ylabel('方位向频谱点数 f_\eta');

%% 方位压缩
Ka = 2*Vr^2*cos(theta_rc)^2./(lambda*Ext_R0_tau_r);
Ha = exp(-1j*pi*Ext_f_eta.^2./Ka);
Offset = exp(-1j*2*pi*Ext_f_eta.*time_eta_c);
S4_tau_feta = S3_tau_feta_RCMC.*Ha.*Offset;
S4_tau_eta = fftshift(ifft(fftshift(S4_tau_feta, 1), Na, 1), 1);

sizec = 32;
c_pos = [168,122];
S4_c = S4_tau_eta(c_pos(1)-sizec/2:c_pos(1)+sizec/2-1, c_pos(2)-sizec/2:c_pos(2)+sizec/2-1);

%% 回波成像
figure('name', "点目标成像结果")
subplot(1, 2, 1);
imagesc(abs(S4_tau_eta));
title('幅度');
xlabel('距离向时间\tau');
ylabel('方位向时间 \eta');
subplot(1, 2, 2);
imagesc(abs(S4_c));
title('幅度');
xlabel('距离向时间\tau');
ylabel('方位向时间 \eta');

%% 目标c切片升采样

Rz = 8;
c_fft = fft2(S4_c, sizec*Rz, sizec*Rz);
figure('name', "目标c频谱")
imagesc(abs(c_fft));
title("目标c频谱");


