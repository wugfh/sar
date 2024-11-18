% 定义参数
f = 5.6e9;  % 载波频率
La = 6;
PRF = 2318;
Tr = 4e-6;
Br = 150e6;
Fr = 200e6;
vr = 7200;
Rc = 600e3;
omega = deg2rad(0.2656);  % 波束旋转速度
theta_c = deg2rad(15);
c = 299792458;

R0 = Rc * cos(theta_c);
lambda = c / f;
Kr = Br / Tr;
Nr = ceil(1.2 * Fr * Tr);

A = 1 - omega * R0 / (vr * cos(theta_c)^2);
Ta = 0.886 * Rc * lambda / (A * La * vr * cos(theta_c));
Na = ceil(1.2*PRF * Ta);
eta_c_strip = -R0 * tan(theta_c) / vr;

% 定义方程并求解
equation = @(x) -R0 * tan(theta_c - omega * x) / vr - x;
etc_c_spot = fsolve(equation, eta_c_strip);  % 解出景中心时间
disp([eta_c_strip, etc_c_spot]);

tau_strip = 2 * sqrt(R0^2 + vr^2 * eta_c_strip^2) / c + (-Nr/2:Nr/2-1) / Fr;
tau_spot = 2 * sqrt(R0^2 + vr^2 * etc_c_spot^2) / c + (-Nr/2:Nr/2-1) / Fr;
eta_strip = eta_c_strip + (-Na/2:Na/2-1) / PRF;
eta_spot = etc_c_spot + (-Na/2:Na/2-1) / PRF;

f_tau = (-Nr/2:Nr/2-1) * Fr / Nr;
feta_c = 2 * vr * sin(theta_c) / lambda;
f_eta = feta_c + (-Na/2:Na/2-1) * PRF / Na;

[mat_tau_strip, mat_eta_strip] = meshgrid(tau_strip, eta_strip);
[mat_tau_spot, mat_eta_spot] = meshgrid(tau_spot, eta_spot);
[mat_ftau, mat_feta] = meshgrid(f_tau, f_eta);

point = [0, 0];

% slide spot
R0_tar = sqrt(point(1)^2 + R0^2);
R_eta_spot = sqrt(R0_tar^2 + (vr * mat_eta_spot - point(2)).^2);
Wr_spot = (abs(mat_tau_spot - 2 * R_eta_spot / c) <= Tr / 2);
Wa_spot = sinc(La * (acos(R0_tar / R_eta_spot) - (theta_c - omega * mat_eta_spot)) / lambda).^2;
Phase_spot = exp(-4j * pi * f * R_eta_spot / c) .* exp(1j * pi * Kr * (mat_tau_spot - 2 * R_eta_spot / c).^2);
S_echo_spot = Wr_spot .* Wa_spot .* Phase_spot;

% strip
R_eta_strip = sqrt(R0_tar^2 + (vr * mat_eta_strip - point(2)).^2);
Wr_strip = (abs(mat_tau_strip - 2 * R_eta_strip / c) <= Tr / 2);
Wa_strip = sinc(La * (acos(R0_tar / R_eta_strip) - theta_c) / lambda).^2;
Phase_strip = exp(-4j * pi * f * R_eta_strip / c) .* exp(1j * pi * Kr * (mat_tau_strip - 2 * R_eta_strip / c).^2);
S_echo_strip = Wr_strip .* Wa_strip .* Phase_strip;

S_tau_feta_spot = fftshift(fft(fftshift(S_echo_spot, 1), Na, 1), 1);
S_tau_feta_strip = fftshift(fft(fftshift(S_echo_strip, 1), Na, 1), 1);

S_ftau_feta_strip = fftshift(fft2(S_echo_strip));
S_ftau_feta_spot = fftshift(fft2(S_echo_spot));

figure(1)
subplot(1, 2, 1)
imagesc(abs(S_tau_feta_spot));
title("slide spotlight")
subplot(1, 2, 2)
imagesc(abs(S_tau_feta_strip));
title("strip")

figure(2)
subplot(1, 2, 1)
imagesc(abs(S_echo_spot));
title("slide spotlight")
subplot(1, 2, 2)
imagesc(abs(S_echo_strip));
title("strip")

figure(3)
subplot(1, 2, 1)
imagesc(abs(S_ftau_feta_spot));
title("slide spotlight")
subplot(1, 2, 2)
imagesc(abs(S_ftau_feta_strip));
title("strip")