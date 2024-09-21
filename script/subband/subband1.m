clear; clc;

%%参数与初始化
EarthMass = 6e24; %地球质量(kg)
EarthRadius = 6.37e6; %地球半径6371km
Gravitational = 6.67e-11; %万有引力常量
f0 = 5.4e9; % 中心频率
H = 755e3; % 飞行高度
Tr = 40e-6; % 脉冲宽度
Br = 2.8e7; % 子带带宽
Fr = 1.2*Br; % 子带采样率
sub_f1 = f0-Br; % 子带1中心频率
sub_f2 = f0; % 子带2中心频率
sub_f3 = f0+Br; % 子带3中心频率
step_f = Br*0.85;
sub_f = [sub_f1;sub_f2;sub_f3];
sub_N = length(sub_f);
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
lambda = c/f0;


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

Nr = ceil(1.2*Fr*Tr);
Na = ceil(1.2*PRF*Ta);

% 构造距离向与方位向时间与频率
t_tau = 2*R_eta_c/c + (-Nr/2:Nr/2-1)*(1/Fr);
t_eta = eta_c+(-Na/2:Na/2-1)*(1/PRF);


f_tau = fftshift((-Nr/2:Nr/2-1)*(Fr/Nr));
f_eta = fnc+(-Na/2:Na/2-1)*(PRF/Na);

[mat_t_tau, mat_t_eta] = meshgrid(t_tau, t_eta);
[mat_f_tau, mat_f_eta] = meshgrid(f_tau, f_eta);


%% 点目标回波产生
pointx = 500;
pointy = 500;
point = [pointx, pointy];

R0_target = sqrt((R0*sin(phi)+point(1))^2+H^2);
R_eta_target = sqrt(R0_target^2+(point(2)-Vr*mat_t_eta).^2);
t_etac_target = (point(2)-R0_target*tan(theta_rc))/Vr;

S_echo = zeros(3, Na, Nr);
% noisy_A = randn([1,sub_N]);
% noisy_P = randn([1,sub_N])*2;
% noisy_T = randn([1,sub_N])*5*1e-9;

noisy_A = zeros(1,sub_N);
noisy_P = zeros(1,sub_N);
noisy_T = zeros(1,sub_N);

for i = 1:sub_N

    mat_t_tau_noisy = mat_t_tau - noisy_T(i);

    Wr = abs(mat_t_tau_noisy-2*R_eta_c/c) < Tr/2;
    % Wa = sinc((La*atan(Vg*(mat_t_eta - t_etac_target)./R0_target)/sub_lambda(i))).^2;
    Wa = abs(mat_t_eta - eta_c)<Ta/2;
    phase = exp(1j*pi*Kr*(mat_t_tau_noisy-2*R_eta_target/c).^2).*exp(-2j*pi*sub_f(i)*2*R_eta_target/c).*exp(1j*pi*noisy_P(i));

    S_echo(i,:,:) = (1+noisy_A(i))*Wr.*Wa.*phase;
end

figure("name", "回波");
for i = 1:sub_N
    subplot(1,sub_N,i);
    sub_S_echo = squeeze(S_echo(i, :, :));
    imagesc(angle(sub_S_echo));
end

%% 距离压缩

figure("name", "各子带距离压缩")
S_ftau_eta = zeros(Na, Nr*sub_N);
t_tau_upsample = 2*R_eta_c/c + (-Nr*sub_N/2:Nr*sub_N/2-1)*(1/(Fr*sub_N));
f_tau_upsample = fftshift((-Nr*sub_N/2:Nr*sub_N/2-1)*(Fr/Nr));
[mat_t_tau_upsample, mat_t_eta_upsample] = meshgrid(t_tau_upsample, t_eta);
[mat_f_tau_upsample, mat_f_eta_upsample] = meshgrid(f_tau_upsample, f_eta);

plot_range = 1:Nr;
for i = 1:sub_N
    sub_S_echo = squeeze(S_echo(i, :, :));
    Hr =  (abs(mat_f_tau)<Br/2).*exp(1j*pi*mat_f_tau.^2/Kr);
    sub_S_ftau_eta = fft(sub_S_echo, Nr, 2);
    sub_S_ftau_eta = sub_S_ftau_eta.*Hr;
    % 频域后置补零，升采样
    tmp = zeros(Na, Nr*sub_N);
    tmp(:, 1:Nr) = sub_S_ftau_eta;
    sub_S_tau_eta = ifft(tmp, Nr*sub_N, 2); 

    % 子带合成
    sub_S_tau_eta = sub_S_tau_eta.*exp(1j*2*pi*(i-(sub_N+1)/2)*step_f.*mat_t_tau_upsample); % 搬移子带频谱

    sub_S_ftau_eta = fft(sub_S_tau_eta, Nr*sub_N, 2);
    S_ftau_eta = S_ftau_eta+sub_S_ftau_eta;

    subplot(1,sub_N,i);
    imagesc(abs(ifft(sub_S_ftau_eta, Nr*sub_N, 2)));

end


S_tau_eta = ifft(S_ftau_eta, Nr*sub_N, 2);


figure("name", "距离压缩");
imagesc(abs(S_tau_eta));

%% RCMC, 相位补偿法
S_tau_feta = fft(S_tau_eta, Na, 1);
delta_R = lambda^2*R0*mat_f_eta_upsample.^2/(8*Vr^2);
G_rcmc = exp(1j*4*pi*mat_f_tau_upsample.*delta_R/c);
S_ftau_feta = fft(S_tau_feta, Nr*sub_N, 2);
S_ftau_feta = S_ftau_feta.*G_rcmc;
S_tau_feta_rcmc = ifft(S_ftau_feta, Nr*sub_N, 2);

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





