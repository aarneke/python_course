"""This code solves the different tasks from the assignment for the Python Course for
Geoscientists at the University of Bremen"""

from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt

class FourierAnalysis:
    def __init__(self, N, noise):

        self.dx = 0.01
        self.L = 2 * np.pi
        self.x_data = np.arange(0, self.L, self.dx)
        self.x_vals = np.linspace(0, self.L, 1000)
        self.n = len(self.x_data)
        self.L_dis = self.n * self.dx
        self.f_data = np.sin(self.x_data) + 0.5 * np.cos(3 * self.x_data)
        self.f_data_noise = np.sin(self.x_data) + 0.5 * np.cos(3 * self.x_data) + 0.2 * np.random.randn(len(self.x_data))
        self.N = N
        self.N_values = list(range(5,N,5))
        self.A = []
        self.B = []
        self.A_dis = []
        self.B_dis = []
        
        if noise:
            self.data = self.f_data_noise
        else:
            self.data = self.f_data

    def f(self,x):
        """Base function for Task 1"""
        return x

    def compute_analytical(self):
        """Calculates the Fourier Coefficients a0, A and B for both tasks"""

        result,_ = integrate.quad(self.f,0,self.L)
        self.a0 = (1/self.L) * result
        self.a0_dis = (1/self.L_dis) * np.sum(self.data) * self.dx

        for n1 in range(1, self.N+1):
            # calculate angular frequency
            omega_n1 = (2 * np.pi * n1) / self.L
            # function for A/B integrals
            integrand_A = lambda x: self.f(x) * np.sin(omega_n1 * x)
            integrand_B = lambda x: self.f(x) * np.cos(omega_n1 * x)

            # Integration
            result_A,_ = integrate.quad(integrand_A, 0, self.L)
            An1 = 2 / self.L * result_A
            self.A.append(An1)

            result_B,_ = integrate.quad(integrand_B, 0, self.L)
            Bn1 = 2 / self.L * result_B
            self.B.append(Bn1)

        for n2 in range(1, self.N + 1):
            omega_n = (2 * np.pi * n2) / self.L_dis
            An2 = (2 / self.L_dis) * np.sum(self.data * np.sin(omega_n * self.x_data)) * self.dx
            Bn2 = (2 / self.L_dis) * np.sum(self.data * np.cos(omega_n * self.x_data)) * self.dx
            self.A_dis.append(An2)
            self.B_dis.append(Bn2)

    def task_1_function(self):
        """Calculates the Fourier approximation for Task 1"""
        f_reconstructed = np.zeros(len(self.x_vals))

        for n in range(1, self.N + 1):  # approximate the function using the coefficients
            omega_n = (2 * np.pi * n) / self.L
            f_reconstructed += self.A[n - 1] * np.sin(omega_n * self.x_vals)  # add the nth sine term
            f_reconstructed += self.B[n - 1] * np.cos(omega_n * self.x_vals)  # add the nth cosine term

        f_reconstructed += self.a0
        return f_reconstructed

    def task_2_function(self):
        """Calculates the discrete Fourier approximation for Task 2"""
        f_dis = np.zeros(len(self.x_data))

        for n in range(1, self.N + 1):  # approximate the function using the discrete coefficients
            omega_n = (2 * np.pi * n) / self.L
            f_dis += self.A_dis[n - 1] * np.sin(omega_n * self.x_data)
            f_dis += self.B_dis[n - 1] * np.cos(omega_n * self.x_data)

        f_dis += self.a0_dis
        return f_dis

    def plot(self,task):
        """Creates the plots for both tasks.
        Use Input argument task = 0 for Task 1 and task ≠ 0 for Task 2"""

        if task == 0:
            x = self.x_vals
            y = self.f(x)
            y_f = f_reconstructed
            name = "Task 1"
        else:
            x = self.x_data
            y = self.data
            y_f = f_dis
            name = "Task 2"

        plt.plot(x, y, label="Original f(x)", color="gray")  # plot the original function
        plt.plot(x, y_f,
                 label=f"Fourier approximation", color="orange")  # plot the approximated function
        plt.legend()
        plt.title(f"Fourier Series Approximation - {name}")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.show()

    # Optional

    def convergence(self):
        """ Displays the convergnece of Fourier approximations for multiple N modes and plots it in a 3D-plot"""

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i, N in enumerate(self.N_values):
            f_reconstructed = np.zeros(len(self.x_vals))

            for n in range(1, N + 1):
                omega_n = (2 * np.pi * n) / self.L
                f_reconstructed += self.A[n - 1] * np.sin(omega_n * self.x_vals)
                f_reconstructed += self.B[n - 1] * np.cos(omega_n * self.x_vals)

            f_reconstructed += self.a0

            ax.plot(self.x_vals, np.full(len(self.x_vals), N), f_reconstructed, label=f"N={N}")

        ax.plot(self.x_vals, 0, self.f(self.x_vals), label="Original", color="gray")
        ax.legend()
        ax.set_xlabel("x")
        ax.set_ylabel("N modes")
        ax.set_zlabel("f(x)")
        ax.set_title("Fourier Series Convergence")
        plt.show()

    def fft(self):
        """Uses the Fast Fourier Transformation - function from numpy and
        plots it together with the manual Fourier approximation from Task 2"""

        fft_result = np.fft.fft(self.f_data)
        frequencies = np.fft.fftfreq(len(self.f_data), d=self.dx)

        n_points = len(self.f_data)

        A_fft = -2 * np.imag(fft_result[1:self.N + 1]) / n_points
        B_fft = 2 * np.real(fft_result[1:self.N + 1]) / n_points

        # compare discrete with fft coefficients
        # print(f"FFT A = {A_fft[0]}, FFT B = {B_fft[0]}, Discrete A = {A_discrete[0]}, Discrete B = {B_discrete[0]}")

        f_fft_reconstructed = np.zeros(len(self.x_data))

        for n in range(1, self.N + 1):
            omega_n = (2 * np.pi * n) / self.L_dis
            f_fft_reconstructed += A_fft[n - 1] * np.sin(omega_n * self.x_data)
            f_fft_reconstructed += B_fft[n - 1] * np.cos(omega_n * self.x_data)

        f_fft_reconstructed += self.a0_dis

        plt.plot(self.x_data, self.data, label="Original", color="gray")
        plt.plot(self.x_data, f_fft_reconstructed, label="FFT reconstruction", color="blue")
        plt.plot(self.x_data, f_dis, label="Manual reconstruction", linestyle="solid", color="orange")
        plt.title(f"Fourier Series Approximation with FFT - Task 2")
        plt.legend()
        plt.show()


# start of the running code

# Boundary conditions
NOISE = True # True = Noise in data
N = 20 # Number of modes

# Set up
fourier = FourierAnalysis(N, NOISE)
fourier.compute_analytical()

# Task 1
f_reconstructed = fourier.task_1_function()
fourier.plot(0) # 0 for Task 1 plot

#Task 2
f_dis = fourier.task_2_function()
fourier.plot(1) # 1 for Task 2 plot

# Optional
fourier.convergence()
fourier.fft()
