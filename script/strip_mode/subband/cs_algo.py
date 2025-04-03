import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

def ista(A, y, lam, max_iter=100, tol=1e-6):
    m, n = A.shape
    x = cp.zeros(n, dtype=cp.complex128)
    L = cp.linalg.norm(A, ord=2)**2  # Lipschitz constant
    for _ in range(max_iter):
        x_old = x.copy()
        x = x + (1 / L) * A.T.conj() @ (y - A @ x)
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
        projections = cp.abs(A.T.conj() @ residual)
        new_index = cp.argmax((projections))
        index_set.append(new_index)
        A[:, new_index] = 0

        A_selected = base[:, index_set]
        ## y = A @ x
        ## 在当前所选基下，求解最接近的x
        x_selected = cp.linalg.pinv(A_selected) @ y 
        
        residual = y - A_selected @ x_selected
    x[index_set] = x_selected
    return x
    

if __name__ == '__main__':
    # Test the OMP algorithm
    Br = 100e6  # Bandwidth of the signal
    T = 10e-6  # Duration of the signal
    Nr = int(1.2*Br * T)  # Number of samples
    k = 2 # Sparsity level
    N = int(cp.ceil(50*k*cp.log(Nr/0.36)))  # Number of measurements
    fs = N/T  # Sampling frequency
    t = cp.linspace(-T/2, T/2, N, endpoint=False)  # Time axis
    Kr = Br/T  # Chirp rate
    x = cp.exp(1j * cp.pi * Kr * t**2) + cp.exp(1j * cp.pi * Kr * (t-2e-6)**2) # Signal
    f = cp.fft.fftshift(cp.arange(-N/2, N/2))*(fs/N) # Frequency axis

    # A = cp.random.randn(N, N)  # Measurement matrix
    # Q,R = cp.linalg.qr(A.T.conj())
    # A = Q.T.conj()
    # y = A @ x
    y = x  # Observed measurements
    print(N, Nr)

    fft_matrix = cp.fft.fft(cp.eye(N), axis=1)/cp.sqrt(N) ## 傅里叶基
    ## match filter
    Hr = cp.exp(1j*cp.pi*f**2/Kr)
    Hr = cp.diag(Hr)
    Trans = fft_matrix.T.conj() @ Hr @ fft_matrix ## 时域的匹配滤波器
    x_match = Trans @ x  ## 匹配滤波
    ## plot
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(t.get(), cp.real(x).get(), label='Original signal')
    plt.xlabel('t')
    plt.ylabel('Amplitude')
    plt.title('Original signal')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(t.get(), cp.abs(x_match).get(), label='Matched signal')
    plt.xlabel('t')
    plt.ylabel('Amplitude')
    plt.title('Matched compressed signal')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../../../fig/subband/cs/match_filter.png')
    ## compressed sensing

    # A_reconvery = A @ Trans.T.conj()
    A_reconvery = Trans.T.conj() ## Trans 是正交的
    # Reconstruct the signal using OMP

    x_reconstructed = ista(A_reconvery, y, 0.1)
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(t.get(), cp.abs(x_reconstructed).get(), label='Reconstructed signal')
    plt.xlabel('t')
    plt.ylabel('Amplitude')
    plt.title('Reconstructed compressed signal')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t.get(), cp.real(x).get(), label='Original signal')
    plt.xlabel('t')
    plt.ylabel('Amplitude')
    plt.title('Original signal abs part')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../../../fig/subband/cs/cs_omp.png')