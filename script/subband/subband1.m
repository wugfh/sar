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
sub_f = f0+(1:sub_N)*step_f;
f0 = sub_f((sub_N+1)/2);

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
lambda = c/(f0);


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
step_T = 0;
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
    Wa = abs(mat_t_eta_offset - eta_c - sub_t_offset(i))<Ta/2;
    sub_phase = exp(1j*pi*Kr*(mat_t_tau_noisy-2*R_eta_target/c).^2).*exp(-2j*pi*sub_f(i)*2*R_eta_target/c).*exp(2j*pi*noisy_P(i));

    S_echo(i,:,:) = (1+noisy_A(i))*Wr.*Wa.*sub_phase;
end


figure("name", "回波");
for i = 1:sub_N
    subplot(1,sub_N,i);
    sub_S_echo = squeeze(S_echo(i, :, :));
    imagesc(angle(sub_S_echo));
end

uprate = sub_N;
Nr_up = Nr*uprate;
Na_up = Na*sub_N;
t_tau_upsample = 2*R_eta_c/c + (-Nr_up/2:Nr_up/2-1)*(1/(Fr*uprate));
f_tau_upsample = fftshift((-Nr_up/2:Nr_up/2-1)*(Fr/Nr));
[mat_t_tau_upsample, mat_t_eta_upsample] = meshgrid(t_tau_upsample, t_eta);
[mat_f_tau_upsample, mat_f_eta_upsample] = meshgrid(f_tau_upsample, f_eta);
S_ftau_eta = zeros(Na, Nr_up);
plot_range = 1:Nr;

figure("name", "子带成像");

target_upsample = zeros(3, Na, Nr_up);

for i = 1:sub_N
    sub_S_echo = squeeze(S_echo(i, :, :));
    Hr =  (abs(mat_f_tau)<Br/2).*exp(1j*pi*mat_f_tau.^2/Kr);
    sub_S_ftau_eta = fft(sub_S_echo, Nr, 2);
    sub_S_ftau_eta = sub_S_ftau_eta.*Hr;
    sub_S_tau_eta = ifft(sub_S_ftau_eta, Nr, 2);

    R_eta = sqrt(R0^2+(Vr*mat_t_eta_offset).^2);
    sub_S_tau_eta = sub_S_tau_eta.*exp(2j*pi*sub_f(i)*2*R_eta/c).*exp(-2j*pi*f0*2*R_eta/c);

    sub_S_tau_feta = fft(sub_S_tau_eta, Na, 1);
    delta_R = lambda^2*R0*mat_f_eta.^2/(8*Vr^2);
    G_rcmc = exp(1j*4*pi*mat_f_tau.*delta_R/c);
    sub_S_ftau_feta = fft(sub_S_tau_feta, Nr, 2);
    sub_S_ftau_feta = sub_S_ftau_feta.*G_rcmc;
    sub_S_tau_feta_rcmc = ifft(sub_S_ftau_feta, Nr, 2);

    mat_R0 = (mat_t_tau*c/2)*cos(theta_rc);
    Ka = 2 * Vr^2 * cos(theta_rc)^2 ./ (lambda * mat_R0);
    sub_S_tau_feta = sub_S_tau_feta_rcmc;
    Ha = exp(-1j*pi*mat_f_eta.^2./Ka);
    offset = exp(-1j*2*pi*mat_f_eta.*eta_c);
    sub_S_tau_feta = sub_S_tau_feta.*Ha.*offset;
    target = ifft(sub_S_tau_feta, Na, 1);

    sub_S_ftau_eta = fft(target, Nr, 2);
    s_f_upsample = zeros(Na, Nr_up);
    s_f_upsample(:,((Nr_up/2-Nr/2):(Nr_up/2+Nr/2-1))) = sub_S_ftau_eta;
    target_upsample(i, :, :) = ifft(s_f_upsample, Nr_up, 2);
    subplot(1,sub_N,i);
    imagesc(abs(target));
end

S_tau_eta = zeros(Na, Nr_up);
R_ref = sqrt(R0_target^2+(point(2)-Vr*(mat_t_eta_upsample+sub_t_offset((sub_N+1)/2))).^2);
for i = 1:sub_N
    target = squeeze(target_upsample(i, :, :));
    R_eta_target = sqrt(R0_target^2+(point(2)-Vr*(mat_t_eta_upsample+sub_t_offset(i))).^2);

    % target = target.*exp(2j*pi*(sub_fnc(i)-fnc).*mat_t_eta_upsample);


    target_shift = target.*exp(2j*pi*(i-(sub_N+1)/2)*step_f.*mat_t_tau_upsample);
    S_tau_eta = target_shift+S_tau_eta;
end

figure("name","最终效果");
imagesc(abs(S_tau_eta));




