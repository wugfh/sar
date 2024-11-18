f = 5.6e9; %载波频率
La = 6;
PRF = 2500;
Tr = 4e-6;
Br = 150e6;
Fr = 200e6;
vr = 7200;
Rc = 600e3;
Ta = 2;
omega = deg2rad(0.05); % 波束旋转速度
theta_c = deg2rad(15);
c = 299792458;

R0 = Rc*cos(theta_c);
lambda = c/f;
Kr = Br/Tr;
Nr = ceil(10*Fr*Tr);

A = 1 - omega*R0/(vr*cos(theta_c)^2);
Na = ceil(PRF*Ta);
eta_c = -R0*tan(theta_c)/vr;


tau = 2*Rc/c + (-Nr/2:Nr/2-1)/Fr;
eta = eta_c+(-Na/2:Na/2-1)/PRF;

f_tau = (-Nr/2:Nr/2-1)*Fr/Nr;
feta_c = 2*vr*sin(theta_c)/lambda;
f_eta = feta_c + (-Na/2:Na/2-1)*PRF/Na;

[mat_tau, mat_eta] = meshgrid(tau, eta);
[mat_ftau, mat_feta] = meshgrid(f_tau, f_eta);



point = [0,100];

% slide spot
R0_tar =  sqrt(point(1)^2+R0^2);
R_eta = sqrt(R0_tar^2+(vr*mat_eta-point(2)).^2);
Wr = (abs(mat_tau-2*R_eta/c) <= Tr/2);
Wa_spot = sinc(La*sin(acos(R0_tar./R_eta)-(theta_c-omega*mat_eta))/lambda).^2;
Phase = exp(-4j*pi*f*R_eta/c).*exp(1j*pi*Kr*(mat_tau-2*R_eta/c).^2);
S_echo_spot = Wr.*Wa_spot.*Phase;

% strip
Wa_strip = sinc(La*sin(acos(R0_tar./R_eta)-theta_c)/lambda).^2;
S_echo_strip = Wr.*Wa_strip.*Phase;

S_tau_feta_spot = fftshift(fft(fftshift(S_echo_spot, 1), Na, 1), 1);
S_tau_feta_strip = fftshift(fft(fftshift(S_echo_strip, 1), Na, 1), 1);
figure(1)
subplot(1,2,1)
imagesc(abs(S_tau_feta_spot));
title("slide spotlight")
subplot(1,2,2)
imagesc(abs(S_tau_feta_strip));
title("strip")

figure(2)
subplot(1,2,1)
imagesc(abs(S_echo_spot));
title("slide spotlight")
subplot(1,2,2)
imagesc(abs(S_echo_strip));
title("strip")