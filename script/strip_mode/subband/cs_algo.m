
% filepath: /G:/sar/script/strip_mode/subband/cs_algo.m
fs = 1000;  % Sampling frequency
T = 1;  % Duration of the signal
t = linspace(0, T, fs);  % Time axis
f1 = 10; f2 = 50;  % Frequencies of the signal
x = exp(2j * pi * f1 * t) + exp(2j * pi * f2 * t);  % Signal
k = 2;  % Sparsity level
N = ceil(20 * k * log(length(x) / 0.2));  % Number of measurements
disp([N, length(x)]);
A = randn(N, length(x));  % Measurement matrix
[Q, R] = qr(A', 0);
A = Q';
y = A * x';  % Observed measurements

fft_matrix = fft(eye(length(x))) / sqrt(length(x));
A_recovery = A / (fft_matrix);
% Reconstruct the signal using OMP
f = (0:length(x)-1) * (fs / length(x));  % Frequency axis
xfft_reconstructed = omp(A_recovery, y, k);
x_fft = fft_matrix * x';
x_reconstructed = (fft_matrix) \ xfft_reconstructed;

figure;
subplot(2, 2, 1);
plot(t, imag(x_reconstructed), 'DisplayName', 'Reconstructed signal');
xlabel('f');
ylabel('Amplitude');
title('Reconstructed imag part');
legend;

subplot(2, 2, 2);
plot(t, imag(x), 'DisplayName', 'Original signal');
xlabel('f');
ylabel('Amplitude');
title('Original signal imag part');
legend;

subplot(2, 2, 3);
plot(t, real(x_reconstructed), 'DisplayName', 'Reconstructed signal');
xlabel('f');
ylabel('Amplitude');
title('Reconstructed signal real part');
legend;

subplot(2, 2, 4);
plot(t, real(x), 'DisplayName', 'Original signal');
xlabel('f');
ylabel('Amplitude');
title('Original signal real part');
legend;

% filepath: /G:/sar/script/strip_mode/subband/cs_algo.m
function x = omp(A, y, k)
    [~, n] = size(A);
    x = zeros(n, 1);
    residual = y;
    index_set = [];
    base = A;

    for i = 1:k
        projections = abs(A' * residual);
        [~, new_index] = max(projections);
        index_set = [index_set, new_index];
        A(:, new_index) = 0;

        A_selected = base(:, index_set);
        x_selected = pinv(A_selected) * y;

        residual = y - A_selected * x_selected;
    end
    x(index_set) = x_selected;
end