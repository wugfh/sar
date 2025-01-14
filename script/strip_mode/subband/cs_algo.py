import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

def ista(A, y, lam, max_iter=100, tol=1e-6):
    m, n = A.shape
    x = cp.zeros(n)
    L = cp.linalg.norm(A, ord=2)**2  # Lipschitz constant
    for _ in range(max_iter):
        x_old = x.copy()
        x = x + (1 / L) * A.T @ (y - A @ x)
        x = cp.sign(x) * cp.maximum(cp.abs(x) - lam / L, 0)
        
        if cp.linalg.norm(x - x_old) < tol:
            break

    return x

def omp(A, y, k):
    m, n = A.shape
    x = cp.zeros(n, dtype=cp.complex128)
    residual = y.copy()
    index_set = []
    base = A.copy()

    for _ in range(k):
        projections = cp.abs(A.T @ residual)
        new_index = cp.argmax((projections))
        index_set.append(new_index)
        A[:, new_index] = 0

        A_selected = base[:, index_set]
        x_selected = cp.linalg.pinv(A_selected) @ y
        
        residual = y - A_selected @ x_selected
    x[index_set] = x_selected
    return x
    

if __name__ == '__main__':
    # Test the OMP algorithm
    fs = 1000  # Sampling frequency
    T = 1  # Duration of the signal
    t = cp.linspace(0, T, fs, endpoint=False)  # Time axis
    f1, f2 = 50, 250  # Frequencies of the signal
    x = cp.cos(2 * cp.pi * f1 * t) + cp.cos(2 * cp.pi * f2 * t)  # Signal
    k = 2 # Sparsity level
    N = int(cp.ceil(20*k*cp.log(len(x)/0.2)))  # Number of measurements
    print(N, len(x))
    A = cp.random.randn(N, len(x))  # Measurement matrix
    Q,R = cp.linalg.qr(A.T)
    A = Q.T
    y = A @ x  # Observed measurements

    fft_matrix = cp.fft.fft(cp.eye(len(x),len(x)), axis=0)
    fft_matrix = cp.ascontiguousarray(fft_matrix)
    A_reconvery = A @ cp.linalg.inv(fft_matrix)
    # Reconstruct the signal using OMP
    f = cp.fft.fftfreq(len(x), 1/fs)  # Frequency axis
    xfft_reconstructed = omp(A_reconvery, y, k*2)
    x_fft = fft_matrix@x
    x_reconstructed = cp.linalg.inv(fft_matrix) @ xfft_reconstructed
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 2, 1)
    plt.plot(cp.abs(xfft_reconstructed).get(), label='Reconstructed signal')
    plt.xlabel('f')
    plt.ylabel('angle')
    plt.title('Reconstructed abs part')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(cp.abs(x_fft).get(), label='Original signal')
    plt.xlabel('f')
    plt.ylabel('angle')
    plt.title('Original signal abs part')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(cp.real(xfft_reconstructed).get(), label='Reconstructed signal')
    plt.xlabel('f')
    plt.ylabel('Amplitude')
    plt.title('Reconstructed signal real part')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(cp.real(x_fft).get(), label='Original signal')
    plt.xlabel('f')
    plt.ylabel('Amplitude')
    plt.title('Original signal real part')
    plt.tight_layout()
    plt.savefig('../../../fig/subband/cs/cs_1d.png')