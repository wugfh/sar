
clear,clc;

len = 4096;
fs = 20*10^(6);
T = 42*10^(-6);
t = (-T/2:1/fs:T/2-1/fs);
[xt, yt] = size(t);
t0 = T/10;
K = -0.41*10^(12);
Bw = abs(K)*T;
alpha = fs/(abs(K)*T);
tbp = abs(K)*(T)^(2);
Kar = kaiser(yt,2.5);
s = exp(pi*1j*K*(t-t0).^2)';
z = zeros(256,1);
ss = [z;s;z;s;z;s;z];
[xs,ys] = size(ss);
ss = [ss;zeros(len-xs,1)];
[xs,ys] = size(ss);
fft_s = fft(s,yt);
h1 = [s;zeros(xs-yt,1)];
H1 = conj(fft(h1,xs));

f = (-fs/2:fs/xs:fs/2-fs/xs);
H2 = (abs(f)<=Bw/2)'.*exp(1j*pi*f.^2/K)';

fft_ss = fft(ss, xs);
out1 = ifft(fft_ss.*H1);
out2 = ifft(fftshift(fft_ss).*H2);


xs=1:xs;
figure(1);
subplot(313);
plot(xs, abs(out2));
title("方式3");
subplot(312);
plot(xs, abs(out1));
title("方式2");
subplot(311);
plot(xs, real(ss));
title("原始信号");




