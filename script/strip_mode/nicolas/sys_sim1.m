clear; clc;

%% 参数

% 轨道参数
c = 299792458;
H = 580e3;
incidence = deg2rad((34.9+41.9)/2);
% incidence = phi;
R_eta_c = H/cos(incidence);
EarthRadius = 6.37e6; % 地球半径6371km
% phi = asin(EarthRadius*sin(incidence)/(EarthRadius+H));
phi = incidence;
R0 = H/cos(incidence);
theta_rc = acos(R0/R_eta_c);

% 卫星参数
Vs = 7560;
Vg = Vs*EarthRadius/(EarthRadius+H);
Vr = sqrt(Vs*Vg);
daz_rx = 1.6;
Naz = 7;
daz_tx = 2;

% 距离向
f0 = 30e9;
lambda = c/f0;
Tr = 5e-6; % 占空比为1.5%
Br = 262e6;
Kr = Br/Tr; 
Fr = 1.2*Br;
Nr = ceil(1.2*Fr*Tr);

% 方位向
B_dop = 0.886*2*Vs*cos(theta_rc)/daz_rx;
Fa = 1350; % PRF
Ta = 0.886*R_eta_c*lambda/(daz_rx*Vg*cos(theta_rc));
Na = ceil(1.2*Fa*Ta);
fnc = 2*Vr*sin(theta_rc)/lambda;
eta_c = -R0*tan(theta_rc)/Vr;

tau = 2*R_eta_c/c + (-Nr/2:Nr/2-1)*(1/Fr);
eta = eta_c+(-Na/2:Na/2-1)*(1/Fa);

f_tau = fftshift((-Nr/2:Nr/2-1)*(Fr/Nr));
f_eta = fnc + fftshift((-Na/2:Na/2-1)*(Fa/Na));

[mat_tau, mat_eta] = meshgrid(tau, eta);
[mat_f_tau, mat_f_eta] = meshgrid(f_tau, f_eta);


%% 生成
point = [0,0];
S_echo = zeros(Naz, Na, Nr);
for i = 1:Naz
    ant_dx = (i-1)*daz_rx;

    R_point = sqrt((R0*sin(phi)+point(1))^2+H^2);
    point_eta_c = (point(2) - R_point*tan(theta_rc))/Vr;
    R_eta_tx = sqrt(R_point^2+(Vr*mat_eta - point(2)).^2);

    mat_eta_rx = mat_eta - ant_dx/Vs;
    R_eta_rx = sqrt(R_point^2+(Vr*mat_eta_rx - point(2)).^2);

    Wr = (abs(mat_tau - (R_eta_tx+R_eta_rx)/c) < Tr/2);
    Wa = (daz_rx*atan(Vg * (mat_eta - point_eta_c) ./ (R0 * sin(phi) + point(1)) / lambda).^2) <= Ta / 2;

    % 多孔径信号的相位，小斜视角近似
    echo_phase_azimuth = exp(-1j*2*pi*f0*(R_eta_rx+R_eta_tx)/c);
    echo_phase_range = exp(1j*pi*Kr*(mat_tau-(R_eta_tx+R_eta_tx)/c).^2);
    S_echo(i,:,:) = Wr.*Wa.*echo_phase_range.*echo_phase_azimuth;
end


%% 成像处理

% 计算重构系数,不同孔径的重构滤波器不一样
P = zeros(Naz, Naz, Na);
H = zeros(Naz, Naz, Na);
prf = Fa;

for k = 1:Naz
    for n = 1:Naz
        % H(k, n, :) = exp(-1j*pi*(Vg/Vs)*((n-1)*daz_rx)^2/(2*lambda*R_point) - 1j*pi*((n-1)*daz_rx)/Vs*(f_eta+(k-1)*prf));
        H(k, n, :) = exp(-1j*pi*((n-1)*daz_rx)/Vs*(f_eta+(k-1)*prf));
    end
end
for j = 1:Na
    H_matrix = squeeze(H(:, :, j));
    tmp = inv(H_matrix);
    % tmp = tmp/(det(tmp)^(1/Naz));
    P(:, :, j) = tmp;
end

% 距离压缩，RCMC
S_r_compress = zeros(Naz, Na, Nr);
for i = 1:Naz
    echo = squeeze(S_echo(i,:,:));
    Hr = (abs(mat_f_tau)<=Br/2).*exp(1j*pi*mat_f_tau.^2/Kr);
    S1_ftau_eta = fft(echo, Nr, 2);
    S1_ftau_eta = S1_ftau_eta.*Hr;
    S1_tau_eta = ifft(S1_ftau_eta, Nr, 2);
    S2_tau_feta = fft(S1_tau_eta, Na, 1);
    S_r_compress(i,:,:) = S2_tau_feta;
end


% 重构

% 需要升采样
uprate = Naz;
f_eta_upsample = fnc + fftshift((-Na*uprate/2:Na*uprate/2-1)*(Fa/Na));
t_eta_upsample = eta_c + (-Na*uprate/2:Na*uprate/2-1)*(1/(Fa*uprate));

[mat_tau_upsample, mat_eta_upsample] = meshgrid(tau, t_eta_upsample);
[mat_f_tau_upsample, mat_f_eta_upsample] = meshgrid(f_tau, f_eta_upsample);

% 计算子带
out_band = zeros(Naz, Na, Nr);
S_out = zeros(Na*uprate, Nr);
for i = 1:Na
    aperture =squeeze(S_r_compress(:, i, :));
    P_aperture = squeeze(P(:,:,i)).';
    tmp = P_aperture*aperture;
    for j = 1:Naz
        out_band(j, i, :) = tmp(j, :);
    end
end

for j = 1:Naz
    % 注意多普勒频率是先正后负的（与速度正相关，靠近为正，远离为负，最短斜距处相对速度最大），
    % 中心频段在最开始。重构后的子带也是这样，需要先把中心频段搬移到中间
    S_out((j-1)*Na+1:j*Na, :) = fftshift(squeeze(out_band(j, :, :)), 1); 
end

% S_ref = zeros(Na*uprate, Nr);
% for j = 1:Naz
%     S_ref((j-1)*Na+1:j*Na, :) = S_tau_feta_rcmc(j, :, :);
% end
S_ref = squeeze(S_r_compress(1, :, :));

figure("name", "rcmc后对比");
subplot(1,2,1);
imagesc(abs(S_out));
title("重构后");

subplot(1,2,2);
imagesc(abs(S_ref));
title("无重构");


delta_R = lambda^2*R_point*mat_f_eta_upsample.^2/(8*Vr^2);
G_rcmc = exp(1j*4*pi*mat_f_tau_upsample.*delta_R/c);
S3_ftau_feta = fft(S_out, Nr, 2);
S3_ftau_feta = S3_ftau_feta.*G_rcmc;
S3_tau_feta_rcmc = ifft(S3_ftau_feta, Nr, 2);

% 方位压缩
mat_R_upsample = mat_tau_upsample*c*cos(theta_rc)/2;
Ka_upsample = 2*Vr^2*cos(theta_rc)^2./(lambda*mat_R_upsample);
Ha_upsample = exp(-1j*pi*mat_f_eta_upsample.^2./Ka_upsample);
Offset_upsample = exp(-1j*2*pi*mat_f_eta_upsample.*eta_c);
out = S3_tau_feta_rcmc.*Ha_upsample.*Offset_upsample;
out = ifft(out, Na*uprate, 1);

% 参考方位压缩
delta_R = lambda^2*R_point*mat_f_eta.^2/(8*Vr^2);
G_rcmc = exp(1j*4*pi*mat_f_tau.*delta_R/c);
S3_ftau_feta = fft(S_ref, Nr, 2);
S3_ftau_feta = S3_ftau_feta.*G_rcmc;
S3_tau_feta_rcmc = ifft(S3_ftau_feta, Nr, 2);

mat_R = mat_tau*c*cos(theta_rc)/2; % 压至零多普勒
Ka = 2*Vr^2*cos(theta_rc)^2./(lambda*mat_R);
Ha = exp(-1j*pi*mat_f_eta.^2./Ka);
Offset = exp(-1j*2*pi*mat_f_eta.*eta_c);

out_ref = S3_tau_feta_rcmc.*Ha.*Offset;
out_ref = ifft(out_ref, Na, 1);


% 显示

figure("name", "成像结果");
subplot(1,2,1);
imagesc(abs(out));
title("重构后成像结果");

subplot(1,2,2);
imagesc(abs(out_ref));
title("无重构成像结果")

[~, r_f_pos] = max(max(abs(S_out), [], 1));
[~, r_pos] = max(max(abs(out), [], 1));

out_f_eta = S_out(:, r_f_pos);
out_eta = out(:, r_pos);
[max_out_eta,a_pos] = max(abs(out_eta));



figure("name", "重构后的切片");
subplot(1,2,1);
plot(f_eta_upsample, abs(out_f_eta));
title("频率切片");

subplot(1,2,2);
plot(t_eta_upsample, 10*log10(abs(out_eta)));
title("成像切片");

[~, r_f_pos_ref] = max(max(abs(S_ref), [], 1));
[~, r_pos_ref] = max(max(abs(out_ref), [], 1));

ref_f_eta = S_ref(:, r_f_pos_ref);
ref_eta = out_ref(:, r_pos_ref);

figure("name", "未重构的切片");
subplot(1,2,1);
plot(f_eta, abs(ref_f_eta));
title("频率切片");

subplot(1,2,2);
plot(eta, abs(ref_eta));
title("成像切片");

figure("name", "各子带频谱");
for i = 1:Naz
    plot(f_eta_upsample((i-1)*Na+1:i*Na), fftshift(abs(out_band(i, :, r_f_pos))), "DisplayName", sprintf("子带%d", i));
    hold on;
end
hold off;
legend('show');
