import matplotlib.pyplot as plt
import numpy as np

class Constants: 
    # in micro meters
    L = 600
    N = 12288
    k0 = 10
    t = 0
    x0 = -100
    c = 3e14
    delta_k = 0.5
    x = np.linspace(-L/2, L/2, N, endpoint=False)


def sum_of_waves(k_lines, A_lines, omega_lines, t):
    # transform from (N,) to (N, 1)
    k = np.asarray(k_lines).ravel()[:, None]
    A = np.asarray(A_lines).ravel()[:, None]
    omega = np.asarray(omega_lines).ravel()[:, None]

    waves = A * np.exp(1j * (k * Constants.x - omega * t))

    # from (1, N) to (N,)
    return waves.sum(axis = 0)
    


def mean_width(values, amplitude):
    """
        Mean and width (standard deviation) of 'values', weighted with |amplitude|^2.
        usage: 'sample_points' and 'complex_amplitudes' are arrays of the same length
        centre, width = mean_width(sample_points, complex_amplitudes)
    """
    P = np.abs(amplitude)**2 / np.sum(np.abs(amplitude)**2)
    mean = np.sum(values * P)
    width = np.sqrt(np.sum((values - mean)**2 * P))
    return mean, width



def plot(x_data, psi_data, k_lines, A_lines, x_lim=(-3, 3)):
    fig, (ax_x, ax_k) = plt.subplots(1, 2, figsize=(11, 4))

    # left panel: real space
    ax_x.plot(x_data, psi_data.real)
    ax_x.set_xlabel(r"$x$ ($\mu$m)")
    ax_x.set_xlim(x_lim)
    ax_x.set_ylabel(r"$\mathrm{Re}\,\psi(x)$")

    # ax_x.plot(x_values, y_values, "o-") # points and line
    # ax_x.plot(k_lines, y_values, "o") # points only
    # ax_x.plot(x_values, y_values, marker="x", markersize=4, linestyle="--")

    # right panel: k-space
    ax_k.stem(k_lines, A_lines.real)
    ax_k.set_xlabel(r"$k$ ($\mu$m$^{-1}$)")
    ax_k.set_ylabel(r"$|A_j|$")
    fig.tight_layout()

    plt.show()


def exercise_1_1(): 
    """
        wave is not localized - it is infinitly long and everywhere A = 1
    """
    A = [1]
    t = 0.0
    k_lines = np.array([Constants.k0])
    A_lines = np.array(A, dtype=complex)
    omega_lines = np.array([Constants.c * k_lines])

    wave_group = sum_of_waves(k_lines, A_lines, omega_lines, t)

    print(wave_group.shape)
    print()

    plot(Constants.x, wave_group, k_lines, A_lines)


def exercise_1_2():
    """

    """
    t = 0.0
    k_lines = np.array([Constants.k0 - Constants.delta_k, Constants.k0 + Constants.delta_k])
    A_lines = np.array([1, 1], dtype=complex)
    omega_lines = Constants.c * k_lines

    wave_group = sum_of_waves(k_lines, A_lines, omega_lines, t)

    plot(Constants.x, wave_group, k_lines, A_lines, x_lim=(-30, 30))

    print("Wave length: ", 2 * np.pi / Constants.k0)
    print("Envelop period: ", 2 * np.pi / Constants.delta_k)


def exercise_1_3():
    """
    As N increases more frequency components interfere, making the central peak narrower and more localized
    """
    t = 0.0
    for exponent in range(5):
        n = 2 ** exponent
        k_lines = Constants.k0 + (np.arange(n) - (n - 1) / 2) * Constants.delta_k
        A_lines = np.ones(n, dtype=complex)
        omega_lines = Constants.c * k_lines

        wave_group = sum_of_waves(k_lines, A_lines, omega_lines, t)

        plot(Constants.x, wave_group, k_lines, A_lines, x_lim=(-300, 300))


def exercise_1_4():
    # 1.4 Gaussian amplitudes
    t = 0.0
    sigma_k = 0.1
    for delta_k in [0.01, 0.05, 0.02]:
        k_lines = np.arange(Constants.k0 - 5 * sigma_k, Constants.k0 + 5 * sigma_k, delta_k)
        A_lines = np.exp(-(k_lines - Constants.k0) ** 2 / (4 * sigma_k ** 2)).astype(complex)
        omega_lines = Constants.c * k_lines

        wave_group = sum_of_waves(k_lines, A_lines, omega_lines, t)

        plot(Constants.x, wave_group, k_lines, A_lines, x_lim=(-300, 300))


def exercise_1_5():
    """
        sigma_x * sigma_k = 0.5 - uncertainty relation
        A small sigma_x requires a large sigma_k, meaning the packet must be 
        composed of a very wide range of different frequencies
    """
    t = 0
    delta_k = 0.01 
    sigma_k_list = [0.25, 0.1, 0.05]

    for sigma_k in sigma_k_list: 
        k_lines = np.arange(Constants.k0 - 5 * sigma_k, Constants.k0 + 5 * sigma_k, delta_k)
        A_lines = np.exp(-(k_lines - Constants.k0) ** 2 / (4 * sigma_k ** 2)).astype(complex)
        omega_lines = Constants.c * k_lines

        wave_group = sum_of_waves(k_lines, A_lines, omega_lines, t)

        _, std_k = mean_width(k_lines, A_lines)

        print("standard deviations k:")
        print("Original {:.2}".format(sigma_k))
        print("Calculated {:.2}".format(std_k))

        _, std_x = mean_width(Constants.x, wave_group)

        print("standard deviations x:")
        print("Calculated {:.2}".format(std_x))

        print("sigma_x * sigma_k: {:.3f}".format(std_k * std_x))

        plot(Constants.x, wave_group, k_lines, A_lines, x_lim=(-300, 300))

        # print(mean_width(k_lines, A_lines))
        # print(mean_width(Constants.x, wave_group))


def exercise_1_6():
    """
        Group velocity:  0.30000000000000004
        Packet speed:  0.3001302083333333
    """
    delta_k = 0.05
    sigma_k = 0.25
    fs_factor = 1e-15

    k_lines = np.arange(Constants.k0 - 5 * sigma_k, Constants.k0 + 5 * sigma_k, delta_k)
    A_lines = np.exp(-(k_lines - Constants.k0) ** 2 / (4 * sigma_k ** 2)).astype(complex)
    omega_lines = Constants.c * k_lines
    peaks_x = []
    wave_groups = []

    mask = (Constants.x >= 0) & (Constants.x <= 2 * np.pi / delta_k)

    for t_fs in [0, 150]:
        t = t_fs * fs_factor
        wave_group = sum_of_waves(k_lines, A_lines, omega_lines, t)
        amplitute = np.abs(wave_group)
        wave_groups.append(wave_group)
        peak = Constants.x[mask][np.argmax(amplitute[mask])]
        peaks_x.append(peak)

    delta_t = 150
    delta_x = np.abs(peaks_x[1] - peaks_x[0])
    print(peaks_x)

    v_dispersion = (omega_lines[-1] - omega_lines[0]) / (k_lines[-1] - k_lines[0]) * fs_factor
    print("Group velocity: ", v_dispersion)

    v_group = delta_x / delta_t
    print("Packet speed: ", v_group)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.plot(k_lines, omega_lines, label="omega(k)")
    ax.set_ylabel(r"$\omega(k)$")
    ax.grid(visible = True)
    ax.legend()
    fig.tight_layout()
    plt.show()

    plot(Constants.x, wave_groups[0], k_lines, A_lines, x_lim=(-300, 300))
    plot(Constants.x, wave_groups[1], k_lines, A_lines, x_lim=(-300, 300))



def exercise_2_1():
    """
        phace velocity v_ph is determined by secant line drawn from (0, 0) to point (k, omega)
        group velocity g_g - tagent line drawn to the dispresion curve at the k0
    """
    w0 = 3.0
    vg = 0.2
    k0 = 10
    sigma_k = 0.1
    delta_k = 0.01
    beta = 0.2 # dispersion

    # k_lines = np.arange(Constants.k0 - 5 * sigma_k, Constants.k0 + 5 * sigma_k, delta_k)
    # A_lines = np.exp(-(k_lines - Constants.k0) ** 2 / (4 * sigma_k ** 2)).astype(complex)
    # omega_lines = w0 + vg * (k_lines - k0) + 0.5 * beta * (k_lines - k0) ** 2

    # wave = sum_of_waves(k_lines, A_lines, omega_lines, 0)

    # plot(Constants.x, wave, k_lines, A_lines)

    def omega(k, b):
        return w0 + vg * (k - k0) + 0.5 * b * (k - k0) ** 2

    beta_list = [0.0, 0.1, 0.2, 1.0]
    k = np.arange(k0 - 5 * sigma_k, k0 + 5 * sigma_k, delta_k)
    v_ph = w0 / k0

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, b in zip(axes.ravel(), beta_list):
        
        ax.plot(k, omega(k, b), "*", label="omega(k)")
        ax.plot(k, v_ph* k, "-", label=f"v_ph = {v_ph:.2f}")
        ax.plot(k, w0 + vg * (k - k0), ":", label=f"v_g = {vg:.2f}")

        ax.set_xlabel(r"$k$ ($\mu$m$^{-1}$)")
        ax.set_ylabel(r"$\omega(k)$")
        ax.grid(visible = True)
        ax.legend()
        ax.set_title(rf"$\beta = {b}$")
    fig.tight_layout()
    plt.show()

    

def exercise_2_2():
    w0 = 3.0
    vg = 0.2
    k0 = 10
    sigma_k = 0.1
    delta_k = 0.01
    beta = 0.0
    c = Constants.c * 1e-15

    k_lines = np.arange(k0 - 5 * sigma_k, k0 + 5 * sigma_k, delta_k)
    A_lines = np.exp(-(k_lines - k0) ** 2 / (4 * sigma_k ** 2)).astype(complex)
    omega_lines = w0 + vg * (k_lines - k0) + 0.5 * beta * (k_lines - k0) ** 2

    vph_pred = w0 / k0
    ng_pred = c / vg
    print("Prediction: vph = {:.4f} um/fs, vg = {:.4f} um/fs, n_g = {:.3f}".format(vph_pred, vg, ng_pred))

    wave = sum_of_waves(k_lines, A_lines, omega_lines, 0)
    plot(Constants.x, wave, k_lines, A_lines)

    def interpolate_complex(x, y, x_point):
        return np.interp(x_point, x, y.real) + 1j * np.interp(x_point, x, y.imag)

    times = np.arange(0, 101, 2.0)  # fs
    group_centres = []
    phase_centres = []
    for t in times:
        wave = sum_of_waves(k_lines, A_lines, omega_lines, t)
        centre, _ = mean_width(Constants.x, wave)
        group_centres.append(centre)
        phase_centres.append(interpolate_complex(Constants.x, wave, centre))

        # print(f"T = {t} {centre}")

    vg_measured, offset = np.polyfit(times, group_centres, 1)

    centre_phase = np.unwrap(np.angle(phase_centres))

    phase_slope, offset = np.polyfit(times, centre_phase, 1)
    omega_phase = k0 * vg_measured - phase_slope
    vph_measured = omega_phase / k0
    n_measured = c / vg_measured

    print("Measurement: vph  = {:.4f} um/fs, vg = {:.4f} um/fs, ng=  {:.4f}".format(vph_measured, vg_measured, n_measured))

    snapshot_times = [0, 20, 40]
    fig, axes = plt.subplots(len(snapshot_times), 1, figsize=(10, 6), sharex=True)
    for ax, t in zip(axes, snapshot_times):
        wave = sum_of_waves(k_lines, A_lines, omega_lines, t)
        ax.plot(Constants.x, wave.real, label="Re-part")
        ax.plot(Constants.x, np.abs(wave), "k--", label="Envelope")
        ax.plot(Constants.x, -np.abs(wave), "k--")
        ax.axvline(vg * t, color="red", alpha=0.7, label="Envelope Centre")
        ax.set_ylabel(f"t = {t} fs")
        ax.set_xlim(-10, 30)
        ax.grid(alpha=0.3)

    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("x")
    fig.tight_layout()
    plt.show()


def exercise_2_3():
    """
        As beta increases the packet evolves over time - widens and decrease flattens by decreasing the amplitude
        DIffrent frequency components travels at diffrent speeds because of beta. 
    """
    w0 = 3.0
    vg = 0.2
    k0 = 10
    betas = [0.05, 0.1, 0.2]
    sigma_k = 0.1
    delta_k = 0.01
    t_max = 600.0
    
    times = [0, 150, 300, 450, 600]

    def interpolate_complex(x, y, x_point):
        return np.interp(x_point, x, y.real) + 1j * np.interp(x_point, x, y.imag)

    for beta in betas: 
        k_lines = np.arange(k0 - 5 * sigma_k, k0 + 5 * sigma_k, delta_k)
        A_lines = np.exp(-(k_lines - k0) ** 2 / (4 * sigma_k ** 2)).astype(complex)
        omega_lines = w0 + vg * (k_lines - k0) + 0.5 * beta * (k_lines - k0) ** 2

        group_centres = []
        phase_centres = []

        fig, axes = plt.subplots(len(times), 1, figsize=(10, 8), sharex=True, sharey=True)
        fig.suptitle(f"Beta = {beta}")

        for ax, t in zip(axes, times):
            wave = sum_of_waves(k_lines, A_lines, omega_lines, t)
            
            ax.plot(Constants.x, wave.real, label="Re(WF))")
            ax.plot(Constants.x, np.abs(wave), "k--", label="Envelope")
            ax.plot(Constants.x, -np.abs(wave), "k--")
            ax.set_ylabel(f"t = {t} fs")
            ax.set_xlim(-20, 160)
            ax.grid(alpha=0.3)

            centre, _ = mean_width(Constants.x, wave)
            group_centres.append(centre)
            phase_centres.append(interpolate_complex(Constants.x, wave, centre))

        axes[0].legend(loc="upper right")
        axes[-1].set_xlabel("x ")
        fig.tight_layout()
        plt.show()

        vg_measured, _ = np.polyfit(times, group_centres, 1)
        centre_phase = np.unwrap(np.angle(phase_centres))

        phase_slope, _ = np.polyfit(times, centre_phase, 1)
        omega_phase = k0 * vg_measured - phase_slope
        vph_measured = omega_phase / k0

        print("Beta {} Measurement: vph  = {:.4f} um/fs, vg = {:.4f} um/fs".format(beta, vph_measured, vg_measured))


def exercise_2_4():
    """
        In x-space as beta increases the packet widens and flattens (decreasing the amplitude).
        In k-space beta steer the omega(k) => -beta/2* (k - k0)^2 = parabolic graph
        with higher beta the value of -beta/2 whcich downward the curves of parabola more steeply
    """
    w0 = 3.0
    vg = 0.2
    k0 = 10.0
    betas = [0.05, 0.1, 0.2]
    sigma_k = 0.1
    delta_k = 0.01
    t_max = 600.0

    k_lines = np.arange(k0 - 5 * sigma_k, k0 + 5 * sigma_k, delta_k)
    A_lines = np.exp(-(k_lines - k0) ** 2 / (4 * sigma_k ** 2)).astype(complex)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for beta in betas:
        omega_lines = w0 + vg * (k_lines - k0) + 0.5 * beta * (k_lines - k0) ** 2
        wave_x = sum_of_waves(k_lines, A_lines, omega_lines, t_max)

        axes[0].plot(Constants.x, np.abs(wave_x),label=f"beta={beta}")

        dispersion_phase = -0.5 * beta * (k_lines - k0) ** 2 * t_max
        axes[1].plot(k_lines, dispersion_phase, label=f"beta={beta}")

    axes[0].set_xlim(-20, 160)
    axes[0].set_xlabel(r"$x$ ($\mu$m)")
    axes[0].set_ylabel(r"$|\psi(x)|$ at $t_{\max}$")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel(r"$k$ ($\mu$m$^{-1}$)")
    axes[1].set_ylabel("Dispersion")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    plt.show()


def exercise_2_5():
    """
        The value of sigma_k * sigma_x = 0.7811 which is > than 0.5 in execise 1.5.
        THis is because sigma_k increases - packet flattens and widen over time.
    """
    w0 = 3.0
    vg = 0.2
    k0 = 10
    sigma_k = 0.1
    delta_k = 0.01
    t_max = 600.0
    beta = 0.1

    k_lines = np.arange(k0 - 5 * sigma_k, k0 + 5 * sigma_k, delta_k)
    A_lines = np.exp(-(k_lines - k0) ** 2 / (4 * sigma_k ** 2)).astype(complex)
    omega_lines = w0 + vg * (k_lines - k0) + 0.5 * beta * (k_lines - k0) ** 2

    wave = sum_of_waves(k_lines, A_lines, omega_lines, t_max)

    _, std_k = mean_width(k_lines, A_lines)
    _, std_x = mean_width(Constants.x, wave)

    print("sigma_k * sigma_x = ", std_k * std_x)

    times = [0, 150, 300, 450, 600]
    fig, axes = plt.subplots(len(times), 1, figsize=(10, 8), sharex=True)

    for ax, t in zip(axes, times):
        w_t = sum_of_waves(k_lines, A_lines, omega_lines, t)
        ax.plot(Constants.x, w_t.real, label="Re(psi)")
        ax.plot(Constants.x, np.abs(w_t), "k--", label="Envelope")
        ax.plot(Constants.x, -np.abs(w_t), "k--")
        ax.set_ylabel(f"t = {t} fs")
        ax.set_xlim(-20, 150)
        ax.grid(alpha=0.3)

    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("x (um)")
    fig.tight_layout()
    plt.show()


def exercise_2_6():
    """
    - Group velocity: Increases linearly with kc
    - Spreading (Width): Does not depend on kc. The packet broadens (envelope width increses 
      and peak amplitude decreases) over time, but this spreading rate is identical for all kc.
    """
    w0 = 3.0
    vg = 0.2
    k0 = 10
    sigma_k = 0.1
    delta_k = 0.01
    t_max = 600.0
    beta = 0.1

    # k_lines_k0 = np.arange(k0 - 5 * sigma_k, k0 + 5 * sigma_k, delta_k)
    # A_lines_k0 = np.exp(-(k_lines_k0 - k0) ** 2 / (4 * k_lines_k0 ** 2)).astype(complex)
    # omega_k0 = w0 + vg * (k_lines_k0 - k0) + 0.5 * beta * (k_lines_k0 - k0) ** 2

    times = np.arange(0, t_max, 200)

    for kc in [9.0, 9.5, 10.0, 10.5]:
        group_centres = []
        widths = []

        for t in times:
            k_lines = np.arange(kc - 5 * sigma_k, kc + 5 * sigma_k, delta_k)
            A_lines = np.exp(-(k_lines - kc) ** 2 / (4 * sigma_k ** 2)).astype(complex)
            omega_lines = w0 + vg * (k_lines - k0) + 0.5 * beta * (k_lines - k0) ** 2

            wave = sum_of_waves(k_lines, A_lines, omega_lines, t)
            centre, width = mean_width(Constants.x, wave)
            group_centres.append(centre)
            widths.append(width)

            # wave_k0 = sum_of_waves(k_lines_k0, A_lines_k0, omega_k0, t)

            # fig, ax_x = plt.subplots(1, 1, figsize=(11, 4))
            # ax_x.plot(Constants.x, wave_k0.real, "-")
            # ax_x.plot(Constants.x, wave.real, "-")
            # ax_x.set_xlabel(r"$x$ ($\mu$m)")
            # ax_x.set_ylabel(r"$\mathrm{Re}\,\psi(x)$")
            # ax_x.set_xlim(-100, 300)
            # fig.tight_layout()
            # plt.show()
        
        vg_measured, offset = np.polyfit(times, group_centres, 1)

        print(f"Kc = {kc}, vg = {vg_measured}")
        print("Width at each t: ", widths)


def exercise_3_1():
    """
        omega(k) and vg(k) repeat every 2*pi/a
        Wavelength at k = pi/a: lambda = 2 * a = 0.5 um
    """
    wc = 3.0
    delta_w = 1.6
    t_max =1000
    a = 0.25 # um

    k0 = 10 
    delta_k = 0.01
    sigma_k = 0.1

    k_lines = np.linspace(-3 * np.pi / a, 3 * np.pi / a, 1000)
    omega = wc - delta_w / 2 * np.cos(k_lines * a)
    vg = delta_w * a * np.sin(k_lines * a) / 2

    wavelength = 2 * np.pi / (np.pi / a)
    print(wavelength)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(k_lines, omega, label=r"$\omega(k)$")
    axes[0].set_ylabel(r"$\omega$")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    
    axes[1].plot(k_lines, vg, color="orange", label=r"$v_g(k)$")
    axes[1].plot(2 * np.pi / a, 0.2, "*")
    axes[1].set_xlabel(r"$k$ ($\mu$m$^{-1}$)")
    axes[1].set_ylabel(r"$v_g$")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    plt.show()


def exercise_3_2():
    wc = 3.0
    delta_w = 1.6
    a = 0.25
    t_max = 1000.0

    kc = np.pi / a
    
    k0 = 10.0
    sigma_k = 0.1
    delta_k = 0.01

    k_lines = np.arange(k0 - 5 * sigma_k, k0 + 5 * sigma_k, delta_k)
    A_lines = np.exp(-(k_lines - k0) ** 2 / (4 * sigma_k ** 2)).astype(complex)
    omega_lines = wc - 0.5 * delta_w * np.cos(k_lines * a)
    
    vg_pred = 0.5 * delta_w * a * np.sin(k0 * a)
    beta_pred = 0.5 * delta_w * (a ** 2) * np.cos(k0 * a)

    print(f"Analytical predictions:")
    print(f"vg = {vg_pred:.4f}")
    print(f"beta = {beta_pred:.4f}")

    times = np.arange(0, t_max + 1, 50.0)
    group_centres = []
    widths = []

    for t in times:
        wave = sum_of_waves(k_lines, A_lines, omega_lines, t)
        centre, width = mean_width(Constants.x, wave)
        group_centres.append(centre)
        widths.append(width)

    group_centres = np.asarray(group_centres)
    widths = np.asarray(widths)

    vg_measured, _ = np.polyfit(times, group_centres, 1)
    print(f"Measured vg = {vg_measured:.6f} um/fs")
    


if __name__ == '__main__':
    exercise_1_1()
    # exercise_1_2()
    # exercise_1_3()
    # exercise_1_4()
    # exercise_1_5()
    # exercise_1_6()
    # exercise_2_1()
    # exercise_2_2()
    # exercise_2_3()
    # exercise_2_4()
    # exercise_2_5()
    # exercise_2_6()
    # exercise_3_1()
    # exercise_3_2()
