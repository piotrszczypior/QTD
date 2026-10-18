import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from scipy.integrate import solve_ivp
from scipy.constants import h


# =============================================================================
# SETUP — parameters and helper functions are given to you
# =============================================================================
# Work through Tasks 1–10 in order. Replace ... with your own code.
# The file is not ready to run all the way through yet, of course.
# You can use exit() after the task you are working on to stop Python there.
# Remove it or move it further down when you continue to the next task.
# The sweeps use if 0: change this to if 1: when you reach them.

wavelength = 1550e-9  # vacuum wavelength (m)
k0 = 2 * np.pi / wavelength
n_air = 1.0
n_core = 3.5
wg_width = 400e-9     # waveguide width (m)
wg_center = -200e-9   # waveguide center (m)
x = np.linspace(-3e-6, 3e-6, 1201)


def plot_mode(x, n, e_vector):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
    axes[0].plot(x * 1e6, n, color="black")
    axes[0].set_ylabel("n(x)")
    axes[1].plot(x * 1e6, np.abs(e_vector)**2, color="orange")
    axes[1].set_xlabel("x (μm)")
    axes[1].set_ylabel("|E(x)|^2 (a.u.)")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    plt.show()


def make_n_step_profile(x, center, width, n_air=n_air, n_core=n_core):
    """Return n_core inside the waveguide and n_air outside."""
    return n_air + (n_core - n_air) * (np.abs(x - center) < width / 2).astype(float)


# =============================================================================
# START HERE — Tasks 1 and 2 are inside solveA
# =============================================================================
def solveA(x, n, k=1, select_mode=0):
    # Input: x and n are arrays, both of length len(x).
    # k: amount of eigenvalues that the solver searches for.
    # select_mode: which mode we return. select_mode=0 gives the TE0 mode.
    # If you want a higher mode, remember to increase k as well.

    # -------------------------------------------------------------------------
    # TASK 1 — Implement A
    # -------------------------------------------------------------------------
    # Define your grid spacing dx and number of interior points N first.
    # Hint: diags can build a sparse matrix from its diagonals.
    # The two boundary points are not part of the eigenproblem.


    dx = 5e-9 # nm - (|-3| + 3) / N 
    N = len(x) - 2

    # calculate_s = lambda : 1 / (k0 * h) ** 2
    # s = calculate_s()
    # d_array = [ni ** 2 - 2 * s for ni in n]
    # print(d_array)
    # diagonals = [d_array, [s for _ in range(N + 2)], [s for _ in range(N + 2)]]
    # A = diags(diagonals, [0, -1, 1])
    # print(A)
    # print(A.shape)

    D2 = diags([1/dx**2, -2/dx**2, 1/dx**2], offsets=[-1, 0, 1], shape=(N,N))
    V = diags(k0**2 * n[1:-1]**2)
    # print(V)

    # print(D2.shape)
    # print(V.shape)
    A = D2 + V
    # print(type(A))

    # Set to 1 to print a part of your matrix and see if it makes sense.
    if 0:
        with np.printoptions(precision=4, suppress=True, linewidth=140):
            print((A * dx**2).toarray()[:6, :6])

    # Show to yourself:
    # - Is A N x N, with N = len(x) - 2?
    # - Do the values within A make sense?
    # - Did you forget the dx**2 factor in the finite differences?

    # -------------------------------------------------------------------------
    # GIVEN — don't touch, just look
    # -------------------------------------------------------------------------
    if True:
        beta_sq_set, e_vector_set = eigsh(A, k=k, which="LA")

        # This is just ordering the modes from neff highest to lowest.
        order = np.argsort(beta_sq_set)[::-1]
        beta_sq_set = beta_sq_set[order]
        e_vector_set = e_vector_set[:, order]

        # This is just choosing the selected mode.
        beta_sq = beta_sq_set[select_mode]
        e_vector = e_vector_set[:, select_mode]

        # Make the largest field value positive, since the returned eigenvectors
        # can be + or - arbitrarily.
        if e_vector[np.argmax(np.abs(e_vector))] < 0:
            e_vector = -e_vector

    # -------------------------------------------------------------------------
    # TASK 2 — Find beta, pad the field and normalize it
    # -------------------------------------------------------------------------
    # Voila, above are beta_sq (=beta^2) (eigenvalue) and the E array (eigenvector)
    # of the selected mode. Now you can normalize and pad to return
    # the desired values.
    # Hint: restore the two boundary values before integrating over x.
    # np.pad might be useful here.


    e_vector_padded = np.pad(e_vector, (1, 1), 'constant', constant_values=(0, 0))
    # print(e_vector_padded)
    # print(beta_sq)

    beta = np.sqrt(beta_sq)
    s = np.trapezoid(np.abs(e_vector_padded)**2, x=x)

    e_vector_norm = e_vector_padded / np.sqrt(s)

    # print(e_vector_norm)

    # Show to yourself:
    # - Is the length of e_vector_norm equal to len(x)?
    # - Are the field values at the two boundaries zero?
    # - Does the following integral give 1? int(np.abs(e_vector_norm)**2)*dx) over all of x
    # If yes, then your mode is normalized! Great job.

    if 0: 
        print("Is the length of e_vector_norm equal to len(x): ", len(x) == len(e_vector_norm))
        integral = np.sum(np.abs(e_vector_norm)**2) * dx
        print("Does the following integral give 1?", int(integral) == 1)

    return beta, e_vector_norm


# =============================================================================
# TASK 3 — Run your code for the first waveguide
# =============================================================================
n1 = make_n_step_profile(x, wg_center, wg_width)  # Here, I give this to you.

# Use solveA to get beta1 and E1, then calculate neff1.
beta1, E1 = solveA(x, n1, k=1, select_mode=0)
print("Beta: ", beta1)
print(E1)
print(len(E1))

neff1 = beta1 / k0
print("N eff1 = ", neff1)

if 0: 
    beta1, E1 = solveA(x, n1, k=5, select_mode=0)
    neff1 = beta1 / k0
    print(f"Mode {0} N eff1 = ", neff1)

    beta1, E1 = solveA(x, n1, k=5, select_mode=1)
    neff1 = beta1 / k0
    print("Mode 1 N eff1 = ", neff1)

    beta1, E1 = solveA(x, n1, k=5, select_mode=2)
    neff1 = beta1 / k0
    print("Mode 2 N eff1 = ", neff1)

    beta1, E1 = solveA(x, n1, k=5, select_mode=3)
    neff1 = beta1 / k0
    print("Mode 3 N eff1 = ", neff1)

    beta1, E1 = solveA(x, n1, k=5, select_mode=4)
    neff1 = beta1 / k0
    print("Mode 4 N eff1 = ", neff1)


plot_mode(x, n1, E1)

# How many guided TE modes are supported? 
# - Supported are: 0 and 1.
# 
# How do the field profiles change with mode order?
# 

# exit()  # Exit just tells Python to stop here; useful while you are working.


# =============================================================================
# TASK 4 — Sweep the waveguide width
# =============================================================================
# Put to 1 when working on this task. Put back to 0 at later tasks if you
# don't want to run this part every single time.
if 0:
    widths = np.linspace(100, 600, 10) * 1e-9
    neff_arr = np.zeros(len(widths))

    for i, w in enumerate(widths):
        # Build the profile for this width and solve its mode.
        # Store the effective index at position i in neff_arr.
        n_w = make_n_step_profile(x, wg_center, w)
        beta_i, _ = solveA(x, n_w)
        neff_arr[i] = beta_i / k0

    # And make a plot please. Uncomment this when your sweep is ready.
    plt.plot(widths * 1e9, neff_arr, "o-", color="tab:blue")
    plt.xlabel("Waveguide width (nm)")
    plt.ylabel(r"Effective index $n_{\mathrm{eff}}$")
    plt.title("Effective index vs waveguide width")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# TASK 5 — What fraction of |E|^2 is outside the first waveguide?
# =============================================================================
# Use n1 and E1 from Task 3. (carefull that your sweep from task 3 doesnt overwrite them).
# Hint: identify the points outside the core, then integrate the intensity
# there. Remember, you already normalized the mode.

if 0: 
    print(E1.shape)
    print(E1)

    left_mask = (x < (wg_center - wg_width / 2))
    right_mask = (x > (wg_center + wg_width / 2))

    left_integral = np.trapezoid(np.abs(E1[left_mask])**2, x=x[left_mask])
    right_integral = np.trapezoid(np.abs(E1[right_mask])**2, x=x[right_mask])

    fraction_outside = left_integral + right_integral
    print(f"Fraction outside for {wg_width} wave guide width: {fraction_outside}")

    widths = np.linspace(100, 600, 10) * 1e-9
    fraction_outside_arr = np.zeros(len(widths))

    for i, w in enumerate(widths):
        # Build the profile for this width and solve its mode.
        # Store the effective index at position i in neff_arr.
        n_w = make_n_step_profile(x, wg_center, w)
        _, E_w = solveA(x, n_w)

        left_mask = (x < (wg_center - w / 2))
        right_mask = (x > (wg_center + w / 2))

        left_integral = np.trapezoid(np.abs(E_w[left_mask])**2, x=x[left_mask])
        right_integral = np.trapezoid(np.abs(E_w[right_mask])**2, x=x[right_mask])
        fraction_outside_arr[i] = left_integral + right_integral

    # And make a plot please. Uncomment this when your sweep is ready.
    plt.plot(widths * 1e9, fraction_outside_arr, "o-", color="tab:blue")
    plt.xlabel("Waveguide width (nm)")
    plt.ylabel(r"Effective index $n_{\mathrm{eff}}$")
    plt.title("Effective index vs waveguide width")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# air = n1 == n_air
# fraction_outside = np.sum(np.abs(E1[air])**2) * 5e-9
# print(fraction_outside)

# =============================================================================
# TASK 6 — Solve the mode of the second waveguide
# =============================================================================
wg2_center = 300e-9


# Build n2, then use your solver to get beta2 and E2.
n2 = make_n_step_profile(x, wg2_center, wg_width)
beta2, E2 = solveA(x, n2)
plot_mode(x, n2, E2)


# =============================================================================
# TASK 7 — Implement the coupling coefficient kappa
# =============================================================================
def calc_kappa(x, beta1, E1, n2, E2):
    # Implement the expression for kappa from the exercise.
    # Remember, you already normalized everything.
    # And you already can do integrals, so should be no problem.
    delta_n_arr = n2 ** 2 - n_air ** 2
    integral = np.trapezoid(np.conjugate(E2) * delta_n_arr * E1, x=x)
    kappa = k0 ** 2 / (2 * beta1) * integral

    return kappa

if 1:
    print("kappa: ", calc_kappa(x, beta1, E1, n2, E2))


# =============================================================================
# TASK 8 — Sweep the gap between the waveguides
# =============================================================================
# Put to 1 when you reach this task.
# We change the gap by moving the center of waveguide 2.
if 0:
    wg2_center_arr = np.linspace(300, 500, 10) * 1e-9
    kappa_arr = np.zeros(len(wg2_center_arr))

    for i, center in enumerate(wg2_center_arr):
        # Rebuild and solve waveguide 2 for each new center.
        # Keep the Task 6 values n2, beta2 and E2 for Task 9; use separate
        # names for the sweep (e.g. n2_i, beta2_i, E2_i).
        n2_i = make_n_step_profile(x, center, wg_width)

        beta2_i, E2_i = solveA(x, n2_i, k=1, select_mode=0)

        # I recommend plotting this so you are sure the gap is changing.
        plot_mode(x, n2_i, E2_i)

        # Use the function calc_kappa and store the result at position i.
        kappa_arr[i] = calc_kappa(x, beta1, E1, n2_i, E2_i)

    # Plot it! Uncomment this when your sweep is ready.
    plt.plot(wg2_center_arr * 1e9, np.abs(kappa_arr), "o-", color="tab:blue")
    plt.xlabel("Waveguide 2 center (nm)")
    plt.ylabel(r"Coupling strength $|\kappa|$ (m$^{-1}$)")
    plt.title("Coupling strength vs waveguide position")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# GIVEN — propagation helper for Task 9
# =============================================================================
# This function is given to you; you don't have to change anything.
def propagate(kappa, beta1, beta2, z):
    delta = (beta1 - beta2) / 2
    M = np.array([
        [delta, kappa],
        [np.conj(kappa), -delta]
    ])
    solution = solve_ivp(
        lambda z, a: 1j * M @ a,
        (z[0], z[-1]),
        [1+0j, 0+0j],  # initially only guide 1 excited
        t_eval=z,
        rtol=1e-8,
        atol=1e-10
    )
    return solution.y  # a1(z), a2(z)


# =============================================================================
# TASK 9 — Propagate and plot the power in both waveguides
# =============================================================================
# Use the modes from Tasks 3 and 6 and your calc_kappa function.
kappa = calc_kappa(x, beta1, E1, n2, E2)

z = np.linspace(0, 50e-6, 1000)
a1, a2 = propagate(kappa, beta1, beta2, z)

# Plot it! Uncomment this when a1 and a2 are ready.
plt.plot(z * 1e3, np.abs(a1)**2, label="Guide 1")
plt.plot(z * 1e3, np.abs(a2)**2, label="Guide 2")
plt.xlabel("z (mm)")
plt.ylabel("Fraction of input power")
plt.legend()
plt.show()


# =============================================================================
# TASK 10 — Reconstruct and plot the field along z
# =============================================================================
# It can be done in one line of code. But feel free to use loops.
# Combine the two mode profiles with their amplitudes at each z.
# Hint: U should have shape (len(z), len(x)).

# U = ...

# Uncomment this when U is ready.
# plt.pcolormesh(x * 1e6, z * 1e3, np.abs(U)**2, shading="auto")
# plt.xlabel("x (µm)")
# plt.ylabel("z (mm)")
# plt.colorbar(label="Field intensity (a.u.)")
# plt.show()
