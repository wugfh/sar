clear; clc;

%%参数与初始化
EarthMass = 6e24; %地球质量(kg)
EarthRadius = 6.37e6; %地球半径6371km
Gravitational = 6.67e-11; %万有引力常量
f0 = 5.4e9; % 中心频率
H = 755e3; % 飞行高度
Tr = 10e-6; % 脉冲宽度
Br = 2.8e7; % 子带带宽
Fr = 1.2*Br; % 子带采样率
step_f = Br;
sub_N = 3;
sub_f = f0:step_f:f0+step_f*sub_N;

% sub_f = [sub_f2];
c = 299792458;
Vr = sqrt(Gravitational*EarthMass/(EarthRadius + H));
Vg = Vr;
Vs = Vr;
La = 15;
phi = deg2rad(20); % 俯仰角
incidence = deg2rad(20.5); % 入射角

Kr = Br/Tr;
sub_lambda = c./sub_f;
lambda = c/(f0+step_f*sub_N/2);


R0 = H/cos(phi);
R_eta_c = H/cos(incidence);
theta_rc = acos(R0/R_eta_c);
eta_c = -R_eta_c*sin(theta_rc)/Vr;

sub_fnc = 2*Vr*sin(theta_rc)./sub_lambda;
fnc = 2*Vr*sin(theta_rc)/lambda;
Ta = 0.886*R_eta_c*lambda/(La*Vg*cos(theta_rc));
delta_fdop = 2*0.886*Vs*cos(theta_rc)/La; 
PRF = 1.7*delta_fdop;
% Ka = 2*Vr^2*cos(theta_rc)^3/(lambda*R0);

Nr = 2*ceil(Fr*Tr);
Na = 2*ceil(PRF*Ta);

% 构造距离向与方位向时间与频率
t_tau = 2*R_eta_c/c + (-Nr/2:Nr/2-1)*(1/Fr);
t_eta = eta_c+(-Na/2:Na/2-1)*(1/PRF);


f_tau = fftshift((-Nr/2:Nr/2-1)*(Fr/Nr));
f_eta = fnc+(-Na/2:Na/2-1)*(PRF/Na);

[mat_t_tau, mat_t_eta] = meshgrid(t_tau, t_eta);
[mat_f_tau, mat_f_eta] = meshgrid(f_tau, f_eta);

% 脉内串发，构造偏移时间
step_T = Tr+Tr;
sub_t_offset = (0:sub_N-1)*step_T; % 添加保护带，偏移时间为2*T

%% 点目标回波产生
pointx = 500;
pointy = 500;
point = [pointx, pointy];



S_echo = zeros(3, Na, Nr);
% noisy_A = randn([1,sub_N]);
% noisy_P = randn([1,sub_N])*2;
% noisy_T = randn([1,sub_N])*5*1e-9;

noisy_A = zeros(1,sub_N);
noisy_P = zeros(1,sub_N);
noisy_T = zeros(1,sub_N); 

for i = 1:sub_N

    mat_t_tau_noisy = mat_t_tau - noisy_T(i);
    mat_t_eta_offset = mat_t_eta+sub_t_offset(i);

    R0_target = sqrt((R0*sin(phi)+point(1))^2+H^2);
    R_eta_target = sqrt(R0_target^2+(point(2)-Vr*mat_t_eta_offset).^2);
    t_etac_target = (point(2)-R0_target*tan(theta_rc))/Vr;

    Wr = abs(mat_t_tau_noisy-2*R_eta_c/c) < Tr/2;
    % Wa = sinc((La*atan(Vg*(mat_t_eta_offset - t_etac_target)./R0_target)/sub_lambda(i))).^2;
    Wa = abs(mat_t_eta_offset - eta_c)<Ta/2;
    phase = exp(1j*pi*Kr*(mat_t_tau_noisy-2*R_eta_target/c).^2).*exp(-2j*pi*sub_f(i)*2*R_eta_target/c).*exp(2j*pi*noisy_P(i));

    S_echo(i,:,:) = (1+noisy_A(i))*Wr.*Wa.*phase;
end


figure("name", "回波");
for i = 1:sub_N
    subplot(1,sub_N,i);
    sub_S_echo = squeeze(S_echo(i, :, :));
    imagesc(angle(sub_S_echo));
end

%% 距离压缩


uprate = sub_N;
Nr_up = Nr*uprate;
Na_up = Na*sub_N;
t_tau_upsample = 2*R_eta_c/c + (-Nr_up/2:Nr_up/2-1)*(1/(Fr*uprate));
f_tau_upsample = fftshift((-Nr_up/2:Nr_up/2-1)*(Fr/Nr));
[mat_t_tau_upsample, mat_t_eta_upsample] = meshgrid(t_tau_upsample, t_eta);
[mat_f_tau_upsample, mat_f_eta_upsample] = meshgrid(f_tau_upsample, f_eta);
S_ftau_eta = zeros(Na, Nr_up);
plot_range = 1:Nr;

R_ref = sqrt(R0_target^2+(point(2)-Vr*(mat_t_eta_upsample+sub_t_offset((sub_N+1)/2))).^2);
for i = 1:sub_N
    sub_S_echo = squeeze(S_echo(i, :, :));
    Hr =  (abs(mat_f_tau)<Br/2).*exp(1j*pi*mat_f_tau.^2/Kr);
    sub_S_ftau_eta = fft(sub_S_echo, Nr, 2);
    sub_S_ftau_eta = sub_S_ftau_eta.*Hr;
    % 频域后置补零，升采样
    tmp = zeros(Na, Nr_up);
    tmp(:, (Nr_up/2):(Nr+Nr_up/2-1)) = fftshift(sub_S_ftau_eta, 2); %对准零频
    % tmp = fftshift(tmp, 2); % 对准f_tau_upsample 频率轴，它的零频在两端
    sub_S_tau_eta = ifft(tmp, Nr_up, 2); 
    % 子带合成
    % imagesc(abs(tmp));
    sub_S_tau_eta = sub_S_tau_eta.*exp(-1j*2*pi*(i-(sub_N+1)/2)*step_f.*(mat_t_tau_upsample)); % 搬移子带频谱
    sub_S_ftau_eta = fft(sub_S_tau_eta, Nr_up, 2);

    R_eta_target = sqrt(R0_target^2+(point(2)-Vr*mat_t_eta_upsample).^2);
    sub_S_ftau_eta = sub_S_ftau_eta.*exp(2j*pi*(i-(sub_N+1)/2)*step_f*2*R_eta_target/c); %补偿跳变相位，此处相位来源于载频不同

    % 补偿方位向时延导致的斜距误差相位
    R1_eta = sqrt(R0_target^2+(point(2)-Vr*(mat_t_eta_upsample+sub_t_offset(i))).^2);
    sub_S_ftau_eta = sub_S_ftau_eta.*exp(2j*pi*2*(R1_eta-R_ref)/lambda);

    %补偿方位向时延
    sub_S_ftau_feta = fft(sub_S_ftau_eta, Na, 1);
    sub_S_ftau_feta = sub_S_ftau_feta.*exp(-2j*pi*(i-(sub_N+1)/2)*step_T.*mat_f_eta_upsample); 
    sub_S_ftau_eta = ifft(sub_S_ftau_feta, Na, 1);

    %合成
    S_ftau_eta = S_ftau_eta+sub_S_ftau_eta;
end

figure("name", "脉压对比");
S_tau_eta = ifft(S_ftau_eta, Nr_up, 2);
range = 1:Nr_up;
S_tau_eta_db = 20*log10(abs(S_tau_eta(round(Na/2),range)));
S_tau_eta_db = (S_tau_eta_db-min(S_tau_eta_db))/(max(S_tau_eta_db)-min(S_tau_eta_db));
sub_S_tau_eta_db = 20*log10(abs(sub_S_tau_eta(round(Na/2),range)));
sub_S_tau_eta_db = (sub_S_tau_eta_db-min(sub_S_tau_eta_db))/(max(sub_S_tau_eta_db)-min(sub_S_tau_eta_db));
plot(range, S_tau_eta_db);
hold on;
plot(range, sub_S_tau_eta_db);
legend("合成带","子带");

figure("name", "合成后的频带");
subplot(1,2,1)
plot(1:Nr_up, abs(S_ftau_eta(round(Na/2),:)));
title("模");
subplot(1,2,2)
plot(1:Nr_up, angle(S_ftau_eta(round(Na/2),:)));
title("相位");
% imagesc(abs(S_ftau_eta));


figure("name", "距离压缩");
imagesc(abs(S_tau_eta));

% Time_tau_Point =  round(((2*R0_target)/(c*cos(theta_rc))-t_tau(1))/(1/(Fr*sub_N)));%目标的距离向标准坐标(距离门)
% Time_eta_Point = round(Na / 2 + (point(2) / Vr) /(1/PRF));
% fprintf("点数坐标应为%d行(方),%d列(距)\n\n",Time_eta_Point,Time_tau_Point);

%% RCMC
% S_tau_feta = fft(S_tau_eta, Na, 1);
% S_tau_feta_rcmc = S_tau_feta;
% IN_N = 8;
% R0_RCMC = t_tau_upsample*c/2;  
% delta_R = lambda^2*f_eta'.^2.*R0_RCMC/(8*Vr^2);
% delta_R_cnt = delta_R*2/(c*(1/Fr));
% for j = 1:Na
%     for k = 1:Nr_up
%         dR = delta_R_cnt(j,k);
%         pos = dR-floor(dR)-(-IN_N/2:IN_N/2-1);
%         rcmc_sinc = sinc(pos);
%         size_sinc = size(rcmc_sinc);
%         predict_value = zeros(size_sinc);
%         for m = -IN_N/2:IN_N/2-1
%             if(k+floor(dR)+m>Nr_up)
%                 predict_value(m+IN_N/2+1) = S_tau_feta(j,k+floor(dR)+m-Nr_up);
%             else
%                 predict_value(m+IN_N/2+1) = S_tau_feta(j,k+floor(dR)+m);
%             end
%         end
%         S_tau_feta_rcmc(j,k) = sum(predict_value.*rcmc_sinc);
%     end
% end

S_tau_feta = fft(S_tau_eta, Na, 1);
delta_R = lambda^2*R0*mat_f_eta_upsample.^2/(8*Vr^2);
G_rcmc = exp(1j*4*pi*mat_f_tau_upsample.*delta_R/c);
S_ftau_feta = fft(S_tau_feta, Nr_up, 2);
S_ftau_feta = S_ftau_feta.*G_rcmc;
S_tau_feta_rcmc = ifft(S_ftau_feta, Nr_up, 2);

figure("name", "rcmc");
imagesc(abs(ifft(S_tau_feta_rcmc, Na, 1)));

%% 方位压缩
mat_R0 = (mat_t_tau_upsample*c/2)*cos(theta_rc);
Ka = 2 * Vr^2 * cos(theta_rc)^2 ./ (lambda * mat_R0);
S_tau_feta = S_tau_feta_rcmc;
Ha = exp(-1j*pi*mat_f_eta_upsample.^2./Ka);
offset = exp(-1j*2*pi*mat_f_eta_upsample.*eta_c);
S_tau_feta = S_tau_feta.*Ha.*offset;
target = ifft(S_tau_feta, Na, 1);

figure("name", "目标");
imagesc(abs(target));




