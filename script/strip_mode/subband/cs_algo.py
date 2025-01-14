import cupy as cp
import matplotlib.pyplot as plt

class CS_Solution:

    def ista(A, y, lam, max_iter=100, tol=1e-6):
        """
        Iterative Shrinkage-Thresholding Algorithm (ISTA) for sparse signal reconstruction.
        
        Parameters:
        A : numpy.ndarray
            Measurement matrix.
        y : numpy.ndarray
            Observed measurements.
        lam : float
            Regularization parameter.
        max_iter : int
            Maximum number of iterations.
        tol : float
            Tolerance for convergence.
        
        Returns:
        x : numpy.ndarray
            Reconstructed sparse signal.
        """
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
        """
        Orthogonal Matching Pursuit (OMP) for sparse signal reconstruction.
        
        Parameters:
        A : numpy.ndarray
            Measurement matrix.
        b : numpy.ndarray
            Observed measurements.
        k : int
            Sparsity level (number of non-zero elements in the reconstructed signal).
        
        Returns:
        x : numpy.ndarray
            Reconstructed sparse signal.
        """
        m, n = A.shape
        x = cp.zeros(n, dtype=cp.complex128)
        residual = y.copy()
        index_set = []
        base = A.copy()

        for _ in range(k):
            projections = cp.abs(A.T @ residual)
            new_index = cp.argmax(cp.abs(projections))
            index_set.append(new_index)
            A[:, new_index] = 0

            A_selected = base[:, index_set]
            x_selected = cp.linalg.lstsq(A_selected, y, rcond=None)[0]
            
            residual = y - A_selected @ x_selected
        x[index_set] = x_selected
        return x
    

if __name__ == '__main__':
    # Test the CS_Solution class
    fs = 1000  # Sampling frequency
    T = 1  # Duration of the signal
    t = cp.linspace(0, T, fs, endpoint=False)  # Time axis
    f1, f2 = 10, 50  # Frequencies of the signal
    x = cp.exp(2j * cp.pi * f1 * t) + cp.exp(2j * cp.pi * f2 * t)  # Signal
    k = 2 # Sparsity level
    N = int(cp.ceil(20*k*cp.log(len(x)/0.2)))  # Number of measurements
    print(N)
    A = cp.random.randn(N, len(x))  # Measurement matrix
    y = A @ x  # Observed measurements

    # Reconstruct the signal using OMP
    x_reconstructed = CS_Solution.omp(A, y, k)
    x_reconstructed = cp.fft.ifft(xfft_reconstructed)
    plt.subplot(2, 1, 1)
    plt.plot(t.get(), cp.real(x).get(), label='Original signal')
    plt.plot(t.get(), cp.real(x_reconstructed).real.get(), label='Reconstructed signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('real part')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(t.get(), cp.imag(x).get(), label='Original signal')
    plt.plot(t.get(), cp.imag(x_reconstructed).get(), label='Reconstructed signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('imaginary part')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../../../fig/subband/cs/cs_1d.png')