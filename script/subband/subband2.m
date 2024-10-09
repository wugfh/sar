clear; clc;

%%参数与初始化
Tr = 0.2e-6; % 脉冲宽度
Br = 2e9; % 子带带宽
Fr = 2.5e9; % 子带采样率
step_f = Br;
sub_N = 3;
sub_f = [33,35,37]*1e9;
f0 = sub_f((sub_N+1)/2);

% sub_f = [sub_f2];
theta_range = deg2rad(8.5);
theta_azimuth = deg2rad(7);
theta_rc = 0;

c = 299792458;
Vr = 5.56; 
Vg = Vr;
Vs = Vr;
phi = deg2rad(25); % 俯仰角
incidence = deg2rad(25); % 入射角

Kr = Br/Tr;
sub_lambda = c./sub_f;
lambda = c/(f0);

La = 0.886*lambda/theta_azimuth;
Lr = 0.886*lambda/theta_range;

R0 = 1.5e2;
H = R0*cos(phi);
R_eta_c = R0/cos(theta_rc);
eta_c = -R_eta_c*sin(theta_rc)/Vr;

sub_fnc = 2*Vr*sin(theta_rc)./sub_lambda;
fnc = 2*Vr*sin(theta_rc)/lambda;
Ta = 0.886*R_eta_c*lambda/(La*Vg*cos(theta_rc));
delta_fdop = 2*0.886*Vs*cos(theta_rc)/La; 
PRF = 1.7*delta_fdop;
% Ka = 2*Vr^2*cos(theta_rc)^3/(lambda*R0);

Nr = ceil(1.5*Fr*Tr);
Na = ceil(1.5*PRF*Ta);

% 构造距离向与方位向时间与频率
t_tau = 2*R_eta_c/c + (-Nr/2:Nr/2-1)*(1/Fr);
t_eta = eta_c+(-Na/2:Na/2-1)*(1/PRF);


f_tau = fftshift((-Nr/2:Nr/2-1)*(Fr/Nr));
f_eta = fnc+fftshift((-Na/2:Na/2-1)*(PRF/Na));

[mat_t_tau, mat_t_eta] = meshgrid(t_tau, t_eta);
[mat_f_tau, mat_f_eta] = meshgrid(f_tau, f_eta);

% 脉内串发，构造偏移时间
step_T = 0;
sub_t_offset = (0:sub_N-1)*step_T; % 添加保护带，偏移时间为step_T

%% 点目标回波产生
pointx = 0;
pointy = 0;
point = [pointx, pointy];

S_echo = zeros(3, Na, Nr);
noisy_A = randn([1,sub_N])*0.1;
noisy_P = randn([1,sub_N])*2;
% noisy_T = randn([1,sub_N])*5*1e-9;

% noisy_A = zeros(1,sub_N);
% noisy_P = zeros(1,sub_N);
noisy_T = zeros(1,sub_N); 
% [noisy_A;noisy_P;noisy_T]

for i = 1:sub_N

    mat_t_tau_noisy = mat_t_tau - noisy_T(i);
    mat_t_eta_offset = mat_t_eta+sub_t_offset(i);

    R0_target = sqrt((R0*sin(phi)+point(1))^2+H^2);
    R_eta_target = sqrt(R0_target^2+(point(2)-Vr*mat_t_eta_offset).^2);
    t_etac_target = (point(2)-R0_target*tan(theta_rc))/Vr;

    Wr = (abs(mat_t_tau_noisy-2*R_eta_c/c) < Tr/2);
    % Wa = sinc((La*atan(Vg*(mat_t_eta_offset - t_etac_target)./R0_target)/sub_lambda(i))).^2;
    Wa = abs(mat_t_eta_offset - eta_c - sub_t_offset(i))<Ta/2;
    sub_phase = exp(1j*pi*Kr*(mat_t_tau_noisy-2*R_eta_target/c).^2).*exp(-2j*pi*sub_f(i)*2*R_eta_target/c);

    S_echo(i,:,:) = (1+noisy_A(i))*Wr.*Wa.*sub_phase.*exp(1j*noisy_P(i));
end


figure("name", "回波");
title("子带回波")
for i = 1:sub_N
    subplot(1,sub_N,i);
    sub_S_echo = squeeze(S_echo(i, :, :));
    imagesc(angle(sub_S_echo));
end

%% 成像

uprate = sub_N*8;
Nr_up = Nr*uprate;
Na_up = Na*sub_N;
t_tau_upsample = 2*R_eta_c/c + (-Nr_up/2:Nr_up/2-1)*(1/(Fr*uprate));
f_tau_upsample = fftshift((-Nr_up/2:Nr_up/2-1)*(Fr/Nr));
[mat_t_tau_upsample, mat_t_eta_upsample] = meshgrid(t_tau_upsample, t_eta);
[mat_f_tau_upsample, mat_f_eta_upsample] = meshgrid(f_tau_upsample, f_eta);

figure("name", "子带成像");

target_upsample = zeros(3, Na, Nr_up);
R_ref = sqrt(R0^2+(Vr*(mat_t_eta+sub_t_offset((sub_N+1)/2))).^2);
for i = 1:sub_N
    sub_S_echo = squeeze(S_echo(i, :, :));
    sub_S_ftau_eta = fft(sub_S_echo, Nr, 2);

    R_eta = sqrt(R0^2+(Vr*(mat_t_eta+sub_t_offset(i))).^2);
    % sub_S_ftau_eta = sub_S_ftau_eta.*exp(4j*pi/c*((sub_f(i)+mat_f_tau).*R_eta-(f0+mat_f_tau).*R_ref));%补偿串发导致的斜距差以及载频不同导致的方位向的误差
    sub_S_tau_eta = ifft(sub_S_ftau_eta, Nr, 2);

    sub_S_tau_eta = sub_S_tau_eta.*exp(2j*pi*(fnc-sub_fnc(i)).*mat_t_eta); %移动多普勒中心，多普勒中心与载频有关系

    target = csa_image(sub_S_tau_eta, Nr, Na, mat_f_eta, R0, R0, fnc, sub_f(i), c, Vr, mat_f_tau, Kr, mat_t_tau);

    subplot(1,sub_N,i);
    imagesc(abs(target));

    sub_S_ftau_eta = fft(target, Nr, 2);
    s_f_upsample = zeros(Na, Nr_up);

    % 不要把0补到频域中间
    [max_value, a_f_pos] = max(max(sub_S_ftau_eta, [], 2));
    [min_value, r_f_pos] = min(sub_S_ftau_eta(a_f_pos, :));
    sub_S_ftau_eta = circshift(sub_S_ftau_eta, -r_f_pos, 2);

    s_f_upsample(:,((Nr_up/2-Nr/2):(Nr_up/2+Nr/2-1))) = sub_S_ftau_eta; 
    target_upsample(i, :, :) = ifft(s_f_upsample, Nr_up, 2);
end

%% 合成
S_ftau_eta = zeros(Na, Nr_up);
echo_ref = squeeze(target_upsample((sub_N+1)/2, :, :));

options = optimoptions('simulannealbnd','PlotFcns',...
          {@saplotbestx,@saplotbestf,@saplotx,@saplotf});  
obj_f = @(coef)syn_and_image(coef, target_upsample, sub_N, Na, Nr_up, mat_f_tau_upsample, mat_t_tau_upsample, Fr, uprate, sub_f);

[x,fval,exitFlag,output] = simulannealbnd(obj_f,[-0.2,0,5],[0.2,0,4],[-4,0,6],options);

coef1 = [noisy_A noisy_P];
coef = x;

for i = 1:sub_N
    target = squeeze(target_upsample(i, :, :));
    target = target./(1+coef(i));
    target = target.*exp(-1j*coef(i+sub_N));

    target_shift = target.*exp(-2j*pi*(f0-sub_f(i)).*mat_t_tau_upsample);

    tar_ftau_eta = fft(target_shift, Nr_up, 2);

    S_ftau_eta = tar_ftau_eta+S_ftau_eta;
end

[~, a_f_pos] = max(max(abs(S_ftau_eta), [], 2));
figure("name", "频域");
subplot(1,2,1);
plot(1:Nr_up, abs(S_ftau_eta(a_f_pos, :)));
title("幅度")
subplot(1,2,2);
plot(1:Nr_up, phase(S_ftau_eta(a_f_pos, :)));
title("相位")

S_tau_eta = ifft(S_ftau_eta, Nr_up, 2);
Svar = var(20*log10(abs(S_tau_eta)));

figure('name', "脉冲对比");

target_one = echo_ref;
% target_one = imrotate(target_one, rad2deg(theta_rc), 'bilinear', 'crop');

[~, a_pos] = max(max(abs(target_one), [], 2));
[~, r_pos] = max(max(abs(target_one), [], 1));
plot_range = r_pos-Nr/2:r_pos+Nr/2;
plot_range = plot_range+(max(1-plot_range(1),1));

target_plot = abs(target_one(a_pos, plot_range));
target_plot = (target_plot-min(target_plot))/(max(target_plot)-min(target_plot));

% S_tau_eta =  imrotate(S_tau_eta, rad2deg(theta_rc), 'bilinear', 'crop');
S_tau_eta_plot = abs(S_tau_eta(a_pos, plot_range));
S_tau_eta_plot = (S_tau_eta_plot-min(S_tau_eta_plot))/(max(S_tau_eta_plot)-min(S_tau_eta_plot));

S_tau_eta_db = 20*log10(S_tau_eta_plot);
target_db = 20*log10(target_plot);


plot(plot_range, S_tau_eta_db);
hold on;
plot(plot_range, target_db);
legend("合成带","子带");
title("子带与合成带距离向剖面");

figure("name","最终效果");
imagesc(abs(S_tau_eta));
title("点目标最终效果");


function y = syn_and_image(coef, subband, sub_N, Na, Nr_up, mat_t_tau_upsample, sub_f)
    S_ftau_eta = zeros(Na, Nr_up);
    f0 = sub_f((sub_N+1)/2);
    for i = 1:sub_N
        target = squeeze(subband(i, :, :));
        target = target./(1+coef(i));
        target = target.*exp(-1j*coef(i+sub_N));
    
        target_shift = target.*exp(-2j*pi*(f0-sub_f(i)).*mat_t_tau_upsample);
    
        tar_ftau_eta = fft(target_shift, Nr_up, 2);
    
        S_ftau_eta = tar_ftau_eta+S_ftau_eta;
    end

    target = S_ftau_eta;
    diff_row = log10(abs(diff(target,1,1))+1);
    diff_col = log10(abs(diff(target,1,2))+1);
    tar = log10(abs(S_ftau_eta)+1);
    y = sum(sum(diff_row))+sum(sum(diff_col))/sum(sum(tar));
end


function image = csa_image(s_echo, Nr, Na, Ext_f_eta, R0, R_ref, fc, f0, c, Vr, Ext_f_tau, Kr, Ext_time_tau_r)
    data_tau_feta = fft(s_echo, Na, 1);

    D = sqrt(1-c^2*Ext_f_eta.^2/(4*Vr^2*f0^2));%徙动因子
    D_ref = sqrt(1-c^2*fc.^2/(4*Vr^2*f0^2)); % 参考频率处的徙动因子（方位向频率中心）
    
    % rcm_total = R0./D - R0/D_ref; % 整体rcm
    rcm_bulk = R_ref./D - R_ref/D_ref; % 一致rcm（参考点的rcm）
    % rcm_diff = rcm_total - rcm_bulk; % 补余rcm（整体rcm与参考rcm的差）
    
    %大斜视角下，距离调频率随距离变化
    K_factor = c*R0*Ext_f_eta.^2./(2*Vr^2*f0^3.*D.^3);
    Km = Kr./(1-Kr*K_factor); 
    
    % delta_tau = 2*rcm_diff/c; % 补余rcmc的变标移动的时间
    
    Ext_echo_tau = Ext_time_tau_r-2*R_ref./(c*D); %以参考目标为中心
    
    s_sc = exp(1j*pi*Km.*(D_ref./D - 1).*Ext_echo_tau.^2);
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
    
    image = ifft(data_tau_feta, Na, 1);
    
end