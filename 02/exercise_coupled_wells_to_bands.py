"""
Exercise: From coupled quantum wells to energy bands (tight binding)

Skeleton code. The helper functions in the first cell are complete and can be
used as they are -- read them carefully, you need to understand what they do.
Everything marked with TODO is your job.
Hand-in: this file with ALL TODOs filled in. Running it must produce all
plots. Write the answers to the questions of the exercise sheet as comments into
the "TODO answer" fields. The file is split into cells (# %%),
which you can run one by one in VS Code, Spyder or PyCharm, or you can copy
them into a Jupyter notebook.

Units used throughout:  z in nm,  energies in eV,  wavefunctions in nm^-1/2.
"""

# %% Imports, parameters and helper functions  (given -- do not change)
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.constants as const
import scipy.sparse as sp
from scipy.linalg import eigh, eigh_tridiagonal

import argparse

# hbar^2 / (2 m_e) in eV nm^2  (~0.0381 eV nm^2)
HBAR2_2M0 = const.hbar**2 / (2 * const.m_e) / const.e * 1e18

M_EFF = 0.067   # effective mass of GaAs in units of m_e (same in well and barrier)
V0 = 0.3        # conduction band offset in eV (wells at -V0, barriers at 0)
W = 5.0         # well width in nm
B = 3.0         # reference barrier width in nm
CLAD = 10.0     # outer barrier (cladding) thickness in nm
DZ = 0.02       # grid spacing in nm


parser = argparse.ArgumentParser()
parser.add_argument("--plot", action="store_true")
args, _ = parser.parse_known_args()


if not args.plot:
    matplotlib.use("Agg")


def show():
    if args.plot:
        plt.show()
    plt.close("all")


def build_potential(layers, dz=DZ):
    """
    Build a 1D heterostructure potential.
    layers: list of (thickness [nm], band edge [eV])
            e.g. [(10, 0.0), (5, -0.3), (10, 0.0)] -> barrier / well / barrier
    returns: z [nm], V [eV] on the interior grid points. The wavefunction is
    zero at z = 0 and z = (len(z)+1)*dz (hard walls at both ends).
    """
    V = np.concatenate([np.full(int(round(d / dz)), Ec, dtype=float)
                        for d, Ec in layers])
    z = (np.arange(len(V)) + 1) * dz
    return z, V


def hamiltonian(z, V, m_eff=M_EFF):
    """
    Finite-difference Hamiltonian H = -hbar^2/(2m) d^2/dz^2 + V as a sparse
    matrix (eV). Use it as  H @ psi  to apply H to a wavefunction.
    """
    dz = z[1] - z[0]
    c = HBAR2_2M0 / m_eff / dz**2
    n = len(z)
    return sp.diags([np.full(n - 1, -c), V + 2 * c, np.full(n - 1, -c)],
                    [-1, 0, 1], format="csr")


def solve_schroedinger(z, V, n_states=1, m_eff=M_EFF):
    """
    Lowest n_states eigenpairs of the 1D Schroedinger equation
        [-hbar^2/(2m) d^2/dz^2 + V(z)] psi = E psi,   psi(0) = psi(L) = 0.
    returns E   [eV]       shape (n_states,)
            psi [nm^-1/2]  shape (len(z), n_states), column i is state i
    psi is normalised (sum |psi|^2 dz = 1) and its left-most lobe is positive.
    """
    dz = z[1] - z[0]
    c = HBAR2_2M0 / m_eff / dz**2
    # the FD Hamiltonian is tridiagonal: diagonal V + 2c, off-diagonals -c
    E, psi = eigh_tridiagonal(V + 2 * c, np.full(len(z) - 1, -c),
                              select="i", select_range=(0, n_states - 1))
    psi = psi / np.sqrt(dz)
    for i in range(n_states):
        first_lobe = np.argmax(np.abs(psi[:, i]) > 0.5 * np.abs(psi[:, i]).max())
        psi[:, i] *= np.sign(psi[first_lobe, i])
    return E, psi


def braket(z, f, g):
    """<f|g> = integral f*(z) g(z) dz on the grid."""
    return np.sum(np.conj(f) * g) * (z[1] - z[0])


# %% Task 1a -- single quantum well
# cladding barrier / well / cladding barrier: (thickness in nm, band edge in eV)
layers_single = [(CLAD, 0.0), (W, -V0), (CLAD, 0.0)]
z, V = build_potential(layers_single)
E, psi = solve_schroedinger(z, V, n_states=2)

# exit()
# print(E)

# TODO: plot V(z) and |psi|^2 of the bound states, shifted to their energies
#       (scale |psi|^2 so that the plot looks nice). Axis labels with units!
# TODO: calculate the decay constant kappa of the ground state in the barrier
#       and check that the cladding CLAD is thick enough.

def calculate_kappa(E0):
    kappa = np.sqrt(M_EFF * (0 - E0) / HBAR2_2M0)
    return kappa

kappa1 = calculate_kappa(E[0])
print("Kappa:", kappa1)

# plot (uncomment and adapt)
# needs: z, V, E, psi of the single well
scale = 0.02                               # scaling of |psi|^2 for the plot
plt.figure(figsize=(6, 4))
plt.plot(z, V, "k", lw=1, label="$E_c(z)$")
for i in range(len(E)):
    plt.plot(z, E[i] + scale * psi[:, i]**2, label=f"$E_{i+1}$ = {E[i]:.3f} eV")
plt.xlabel("z (nm)")
plt.ylabel("E (eV)")
plt.title("Single quantum well")
plt.legend()
show()

# TODO answer (Is the cladding thick enough? What goes wrong with 1 nm?):
# Yes, cladding is thick enough. The wavefunction decays into the walls exponentially so that it becomes insignificant
# The boundary for CLAD = 1nm is too close to the well, which may lead to error in energy calculation.
# For CLAD == 10nm ==> the kappa = 0.608 nm^-1
# For CLAD == 1nm  ==> the kappa = 0.556 nm^-1


# %% Task 1b -- double quantum well (barrier width B)
# TODO: write the layer list of the double well by hand (two wells of width W,
#       separated by a barrier of width B, cladding CLAD on both sides),
#       solve for the two lowest states (E2, psi2 on z2, V2) and plot psi
#       (not |psi|^2!) together with the potential and the single-well E_0.
E0_single = E[0]
layers_double = [(CLAD, 0.0), (W, -V0), (3.0, 0.0), (W, -V0), (CLAD, 0.0)]
z2, V2 = build_potential(layers_double, dz=DZ)
E2, psi2 = solve_schroedinger(z2, V2, n_states=2)

# print(z2.shape)
# print(V2.shape)
# print(E2)
# print(z)
# print(V)


# plot (uncomment and adapt)
# needs: z2, V2, E2, psi2 of the double well, E0_single = E[0] of Task 1a
scale = 0.05                               # scaling of psi for the plot
plt.figure(figsize=(6, 4))
plt.plot(z2, V2, "k", lw=1)
for i, label in enumerate(["state 1", "state 2"]):
    plt.plot(z2, E2[i] + scale * psi2[:, i], label=label)
plt.axhline(E0_single, color="gray", ls=":", label="single well $E_0$")
plt.xlabel("z (nm)")
plt.ylabel("E (eV)")
plt.ylim(-V0 - 0.02, 0.02)
plt.legend()
show()

# TODO answer (symmetry, bonding/antibonding, position relative to E_0):
# The first ground state is symetric and bonding
# The second is asymetric and antibonding.
# Both states are equally apart from E0.
# The antibonding state has higher energy and has a node between wells.


# %% Task 1c -- splitting vs. barrier width
barriers = np.arange(0.5, 6.01, 0.25)
dE_num = []

# TODO: loop over the barrier widths (build the layer list inside the loop),
#       store E_2 - E_1, plot on a log scale
#       and compare the slope with kappa.

for barrier in barriers: 
    layers = [(CLAD, 0.0), (W, -V0), (barrier, 0.0), (W, -V0), (CLAD, 0.0)]
    z, V = build_potential(layers, dz=0.1)
    E, _ = solve_schroedinger(z, V, n_states=2)
    dE_num.append(E[1] - E[0])

# plot (uncomment and adapt)
# needs: barriers, dE_num (in eV), kappa
dE_num = np.array(dE_num)
plt.figure(figsize=(6, 4))
plt.semilogy(barriers, 1e3 * dE_num, "o", label="numerical")
plt.semilogy(barriers, 1e3 * dE_num[-1] * np.exp(-kappa1 * (barriers - barriers[-1])),
             ":", label=r"$\propto e^{-\kappa b}$")
plt.xlabel("barrier width b (nm)")
plt.ylabel(r"$\Delta E$ (meV)")
# plt.yscale('log', base=2)
plt.legend()
show()



# TODO: 
# print(1e3 * dE_num[-1] * np.exp(-kappa1 * (barriers - barriers[-1])))

print(np.polyfit(barriers, np.log(dE_num), 1))

# TODO answer (functional form, slope vs. kappa):
# The splitting decreases exponetially with increasing width of barrier between wells. 
# The slope component of is ~ -kappa. The wider the barrier the less likely the wavefunction will tunnel between wells. 


# %% Task 2a -- isolated well states and matrix elements (barrier width B)
# From now on we need structures with many wells, and the single wells
# "alone" on exactly the same grid as the full structure. This helper
# (given) builds the layer list for you.

def superlattice_layers(n_wells, w=W, b=B, v0=V0, clad=CLAD, active=None):
    """
    Layer list for n_wells wells of width w separated by barriers of width b,
    surrounded by cladding barriers of thickness clad.
    active: indices of the wells that are actually present (default: all).
            Missing wells are filled with barrier material, so the grid stays
            identical -> use this to get the isolated well states phi_j.
    Example: superlattice_layers(2, active=[0]) -> only the left well of a
             double-well structure, on exactly the same grid.
    """
    if active is None:
        active = range(n_wells)
    layers = [(clad, 0.0)]
    for j in range(n_wells):
        layers.append((w, -v0 if j in active else 0.0))
        if j < n_wells - 1:
            layers.append((b, 0.0))
    layers.append((clad, 0.0))
    return layers


z, V = build_potential(superlattice_layers(2))
# TODO: phi_L, phi_R = ground states of the left / right well alone
#       (use the `active` argument), E0 = their energy
# TODO: plot phi_L and phi_R together with the double-well potential V

z_left, V_left = build_potential(superlattice_layers(2, active=[0]))
z_right, V_right = build_potential(superlattice_layers(2, active=[1]))

E_left, phi_L = solve_schroedinger(z_left, V_left)
_, phi_R = solve_schroedinger(z_right, V_right)

E0 = E_left[0]


# plot (uncomment and adapt)
# needs: z, V, phi_L, phi_R, E0
scale = 0.05
plt.figure(figsize=(6, 4))
plt.plot(z, V, "k", lw=1)
plt.plot(z, E0 + scale * phi_L, label=r"$\varphi_L$")
plt.plot(z, E0 + scale * phi_R, label=r"$\varphi_R$")
plt.xlabel("z (nm)")
plt.ylabel("E (eV)")
plt.ylim(-V0 - 0.02, 0.02)
plt.legend()
show()


# %% Task 2b -- matrix elements
# TODO: H = hamiltonian(z, V)           # Hamiltonian of the full double well
# TODO: eps = <phi_L|H|phi_L>,  t = -<phi_L|H|phi_R>,  r = <phi_L|phi_R>
#       (use braket and H @ phi)

H = hamiltonian(z, V) ## TODO: read
eps = braket(z, phi_L, H @ phi_L)
t = - braket(z, phi_L, H @ phi_R)
r = braket(z, phi_L, phi_R)

print("eps: ", eps)
print("t: ", t)
print("r: ", r)
print("E0", E0)

print("eps - E0:", eps - E0)

# TODO answer (eps vs. E0):
# The diffrence between eps and E0 is ~ 0.000461. The epsilon is a little bit lower than the isolated well
# because the second well introduces aditional potencial. The second wavefunction reaches the second well, 
# decreasing slightly the energy

# %% Task 2c -- solve the 2x2 problem with and without overlap
# TODO: H_2x2 = [[eps, -t], [-t, eps]],  S = [[1, r], [r, 1]]
# TODO: r != 0: E_S, c_S = eigh(H_2x2, S)
#       r  = 0: E_0r, c_0r = eigh(H_2x2)
# TODO: check E_S against E_pm = (eps -+ t)/(1 +- r) and compare both
#       with the exact energies of Task 1b

def solve_lcao(H=H, eps=eps, t=t, r=r):
    H_2x2 = [[eps, -t], [-t, eps]]
    S = [[1, r], [r, 1]]
    return eigh(H_2x2, S)

E_S, c_S = solve_lcao()
E_0r, c_0r = solve_lcao(r=0)

E_plus, E_minus = (eps - t) / (1 + r), (eps + t) / (1 - r)

print("With overlap:", E_S)
print("Without overlap:", E_0r)
print("Analyticaly:", E_plus, E_minus)
print("Exact energies:", E2)

# The energies agree with the eigevalue problem when the overlap is included


# print(E_S)
# print("c_S:",c_S)
# print("PHI", phi_L)
# print(c_0r)
# print(E_S)
# print(E_0r)


# %% Task 2d -- reconstruction psi = c_L phi_L + c_R phi_R
# TODO: reconstruct both states for r != 0 (psi_S) and r = 0 (psi_0r), plot |psi|^2 on top of
#       the exact states of Task 1b
# TODO: print <psi|psi> and <psi|H|psi>/<psi|psi> for every reconstructed state

psi_S = c_S[0] * phi_L + c_S[1] * phi_R
psi_0r = c_0r[0] * phi_L + c_0r[1] * phi_R

# plot (uncomment and adapt)
# needs: z, V, E2, psi2 of Task 1b, E_S, E_0r and the reconstructed states
#        psi_S, psi_0r (columns = states, like psi2)
scale = 0.04
fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, E_l, psi_l, title in [(axs[0], E_S, psi_S, r"LCAO, $r \neq 0$"),
                              (axs[1], E_0r, psi_0r, "LCAO, $r = 0$")]:
    ax.plot(z, V, "k", lw=1)
    for i in range(2):
        ax.plot(z, E2[i] + scale * psi2[:, i]**2, "-", color=f"C{i}",
                label=f"exact, state {i+1}")
        ax.plot(z, E_l[i] + scale * psi_l[:, i]**2, "--", color=f"C{i}",
                label=f"LCAO, state {i+1}")
    ax.set_title(title)
    ax.set_xlabel("z (nm)")
    ax.set_ylim(E2[0] - 0.03, E2[1] + 0.03)
axs[0].set_ylabel("E (eV)")
axs[0].legend(fontsize=8)
show()


def print_norm_and_expectation_value(psi, z=z, H=H):
    for i in range(2): # 2 states
        i_state_p = psi[:, i]
        norm = braket(z, i_state_p, i_state_p)
        energy_expect_value = braket(z, i_state_p, H @ i_state_p) / norm

        print(" State: {}, norm: {:.4f}, Energy expectation value: {:.4f}".format(i, norm, energy_expect_value))

print("PSI_S: r != 0")
print_norm_and_expectation_value(psi=psi_S)

print("PSI_0r: r == 0")
print_norm_and_expectation_value(psi=psi_0r)


# TODO answer (meaning of (c_L, c_R); difference r = 0 vs. r != 0: shape, norm or energy?; role of S):
# The sign of the second component tells the symmetry
# The overlap affects the normalization therefore without the S Matrix the calculations give wrong eigenvalues but the same state shapes

# %% Task 2e -- splitting with and without overlap
# TODO: repeat 2a-2c inside the loop of Task 1c (store dE_S, dE_0r) and add both LCAO
#       splittings to the plot of Task 1c

def calculate_splitting():
    dE_S = []
    dE_0r = []
    _barriers = np.arange(0.5, 6.01, 0.25)

    for barrier in _barriers: 
        z, V = build_potential(superlattice_layers(2, b=barrier))
        E0, _ = solve_schroedinger(z, V)

        z_left, V_left = build_potential(superlattice_layers(2, b=barrier, active=[0]))
        z_right, V_right = build_potential(superlattice_layers(2, b=barrier, active=[1]))

        _, phi_L = solve_schroedinger(z_left, V_left)
        _, phi_R = solve_schroedinger(z_right, V_right)

        H = hamiltonian(z, V)
        eps = braket(z, phi_L, H @ phi_L)
        t = - braket(z, phi_L, H @ phi_R)
        r = braket(z, phi_L, phi_R)

        E_S, c_S = solve_lcao(H=H, eps=eps, t=t, r=r)
        E_0r, c_0r = solve_lcao(H=H, eps=eps, t=t, r=0)

        dE_S.append(E_S[1] - E_S[0])
        dE_0r.append(E_0r[1] - E_0r[0])

    return dE_S, dE_0r


dE_S, dE_0r = calculate_splitting()

# plot (uncomment and adapt)
# needs: barriers, dE_num, dE_S, dE_0r (in eV)
plt.figure(figsize=(6, 4))
plt.semilogy(barriers, 1e3 * np.array(dE_num), "o", label="numerical (Task 1c)")
plt.semilogy(barriers, 1e3 * np.array(dE_S), "-", label=r"LCAO, $r \neq 0$")
plt.semilogy(barriers, 1e3 * np.array(dE_0r), "--", label="LCAO, $r = 0$")
plt.xlabel("barrier width b (nm)")
plt.ylabel(r"$\Delta E$ (meV)")
plt.legend()
show()

# plt.figure(figsize=(6, 4))
# plt.plot(barriers, np.array(dE_0r) / np.array(dE_S))
# plt.xlabel("barrier width b (nm)")
# plt.ylabel(r"$\Delta E_{r=0} / \Delta E_{r\neq 0}$")
# show()

# TODO answer (is r = 0 better for wide barriers?):
# No, the slope state consistant and the gap in energies does not drop as barrier width increases



plt.figure(figsize=(6, 4))

# %% Task 3a -- from one to many wells
for N in range(1, 11):
    z, V = build_potential(superlattice_layers(N))
    E_N, _ = solve_schroedinger(z, V, n_states=N)

    pass  # TODO: solve the N-well structure for the lowest N states, plot E vs N

    # plot (uncomment and adapt)
    # needs: E_N = energies of the N-well structure
    # note:  put plt.figure before the loop and the plt.plot line inside the loop
    plt.plot(np.full(N, N), E_N, "_", ms=14, mew=1.5, color="C0")

plt.xlabel("number of wells N")
plt.ylabel("E (eV)")
show()

# TODO answer (number of states, band centre and width vs. N, N -> infinity):
# As the number of wells N increases, the single energy level splits into N different states
# that become closer together
# The center of energy band stays mostly consistent 
# When the N -> inf then all energy levels would create a consistent band 


# %% Task 3b -- wavefunctions of the miniband (N = 6)
# TODO: plot all six states in six panels below each other

N_WELLS = 6
z6, V6 = build_potential(superlattice_layers(N_WELLS))
_, psi6 = solve_schroedinger(z6, V6, n_states=N_WELLS)

# plot (uncomment and adapt)
# needs: z6, V6, psi6 of the 6-well structure
fig, axs = plt.subplots(6, 1, figsize=(6, 8), sharex=True)
for i in range(6):
    axs[i].plot(z6, psi6[:, i])
    axs[i].plot(z6, V6, "k", lw=0.5)
    axs[i].set_ylabel(f"n = {i+1}")
axs[-1].set_xlabel("z (nm)")
show()

# TODO answer (envelope, number of nodes, particle in a box):
# The amplitude of wavefunction chages with the n value. 
# the number of nodes increases with n value - similar to particle in the box where higher energy states have more nodes. 



# %% Task 3c -- bandwidth vs. barrier width (N = 20)
# TODO: loop over b = 1 ... 6 nm, plot the bandwidth (log scale) together
#       with the double-well splitting of Task 1c


N_WELLS = 20
barrier_widths = np.arange(1.0, 6.01, 0.25)
bandwidths = []

for barrier in barrier_widths:
    z20, V20 = build_potential(superlattice_layers(N_WELLS, b=barrier))
    E, _ = solve_schroedinger(z20, V20, n_states=N_WELLS)
    bandwidths.append(E[-1] - E[0])


# plot (uncomment and adapt)
# needs: bs, bw (band widths in eV), barriers and dE_num of Task 1c
plt.figure(figsize=(6, 4))
plt.semilogy(barrier_widths, 1e3 * np.array(bandwidths), "o", label="band width, N = 20")
plt.semilogy(barriers, 1e3 * np.array(dE_num), "-", label="double-well splitting")
plt.xlabel("barrier width b (nm)")
plt.ylabel("energy (meV)")
plt.legend()
show()

# TODO answer (bandwidth vs. b, comparison with the splitting, design):
# As the barriers get wider, the bandwidth gets smaller because the wells are less connected.
# The double-well splitting also decreases. 
# Thin barriers give a wider miniband, while thick barriers give a narrower



# %% Task 4a -- E(k) with eps, t, r from Task 2
a = W + B
# TODO: E(k) = (eps - 2 t cos(k a)) / (1 + 2 r cos(k a)) for -pi/a <= k <= pi/a;
#       plot it
# TODO: band width from E(k)

k = np.linspace(-np.pi / a, np.pi / a, 200)
E_k = (eps - 2 * t * np.cos(k * a)) / (1 + 2 * r * np.cos(k * a))

# plot (uncomment and adapt)
# needs: k (1/nm), E_k (eV)
plt.figure(figsize=(6, 4))
plt.plot(k * a / np.pi, E_k, label="tight binding")
plt.xlabel(r"$k a / \pi$")
plt.ylabel("E (eV)")
plt.legend()
show()

bandwidth = E_k.max() - E_k.min()
print("Bamdwidth:", bandwidth)

# TODO answer (band width from E(k) vs. Task 3c):
# The bandwidth is closed to the one calculated for 20 wells in Task 3c


# %% Task 4b -- numerical states of Task 3 (N = 20) in the E-k diagram
# TODO: k_n = n pi / ((N+1) a); plot the energies of the 20-well structure at
#       +k_n and -k_n on top of E(k) of Task 4a (b = B)


N_WELLS = 20
n = np.arange(1, N_WELLS + 1)

z20, V20 = build_potential(superlattice_layers(N_WELLS, b=3.0))
E_20, _ = solve_schroedinger(z20, V20, n_states=N_WELLS)
k_n = np.pi * n / ((N_WELLS + 1) * a)


# plot (uncomment and adapt)
# needs: k, E_k of Task 4a, k_n, E_20 (energies of the 20-well structure)
# note:  Task 4c adds the parabola to this figure
plt.figure(figsize=(6, 4))
plt.plot(k * a / np.pi, E_k, label="tight binding")
plt.plot(k_n * a / np.pi, E_20, "o", color="C2", label="Schroedinger, N = 20")
plt.plot(-k_n * a / np.pi, E_20, "o", mfc="none", color="C2")
plt.xlabel(r"$k a / \pi$")
plt.ylabel("E (eV)")
plt.legend()
show()

# TODO answer (choice of k_n, N -> infinity):

# We choose these k values because the wavefunctions behave ike standing waves, as we saw in Task 3b.
# Higher states have more nodes, so they have larger k values.
# When N becomes very large, the allowed k values get closer and closer together and eventually form a continuous band



# %% Task 4c -- effective mass
def fit_parabola(k, E):
    """
    Fit E = E_c + hbar^2 k^2 / (2 m*) to the points (k, E)  (given).
    k: wave vectors [1/nm], E: energies [eV]
    returns E_c [eV], m_star [in units of m_e]
    """
    slope, E_c = np.polyfit(k**2, E, 1)     # straight line in k^2
    return E_c, HBAR2_2M0 / slope


# TODO: select the numerical states with |k| a / pi <= 0.5 and fit them with
#       fit_parabola; draw E_c + HBAR2_2M0 / m_star * k**2 in the same E-k
#       diagram as Task 4b and compare m_star with M_EFF

mask = k_n * a / np.pi <= 0.5
E_c, m_star = fit_parabola(k_n[mask], E_20[mask])


# plot (uncomment and adapt)
# needs: k, E_k, k_n, E_20 of Task 4b and E_c, m_star of the fit
plt.figure(figsize=(6, 4))
plt.plot(k * a / np.pi, E_k, label="tight binding")
plt.plot(k_n * a / np.pi, E_20, "o", color="C2", label="Schroedinger, N = 20")
plt.plot(-k_n * a / np.pi, E_20, "o", mfc="none", color="C2")
plt.plot(k * a / np.pi, E_c + HBAR2_2M0 / m_star * k**2, ":", color="C3",
         label=f"parabola, m* = {m_star:.3f} m_e")
plt.axvspan(-0.5, 0.5, color="gray", alpha=0.1)   # fit range
plt.ylim(E_20.min() - 0.01, E_20.max() + 0.01)
plt.xlabel(r"$k a / \pi$")
plt.ylabel("E (eV)")
plt.legend()
show()

# TODO answer (why heavier than bulk, m* vs. barrier width, validity of the parabola, m* near k = pi/a):
# The effective mass is larger than the bulk mass because the barriers make it harder for electrons to move.
# If the barriers get wider, the effective mass gets bigger.
# The parabola is a good approximation near k = 0, but it doesn't work perfectly for larger k values.
# Near the edge of the band, the curvature changes sign, so the effective mass becomes negative


# %% Task 4d -- discussion
# TODO answer (mapping to a real crystal, validity of nearest-neighbour TB):
# The quantum wells are similar to atoms in a real crystal, and the wavefunctions phi_j are like atomic orbitals.
# The parameter t describes how  electrons can jump to neighbouring wells.
# Tight binding works best when the wells are not too strongly coupled, so that only the neighbouring wells 
# interact with each other
# For bigger barriers, the tunneling is weaker, which makes the tight-binding approximation better and the band narrower


# %% Bonus -- density of states
# TODO: Z(E) from the tight-binding E(k), two-panel plot E(k) | Z(E),
#       histogram of the eigenvalues of a 200-well structure

N_WELLS = 200
n = np.arange(1, N_WELLS + 1)
L = N_WELLS * a 

Z_E = 1 / np.pi * np.abs(np.gradient(E_k, k)) ** -1
z200, V200 = build_potential(superlattice_layers(N_WELLS, b=3.0))
E_200, _ = solve_schroedinger(z200, V200, n_states=N_WELLS)


# plot (uncomment and adapt)
# needs: k, E_k (E(k) for 0 <= k <= pi/a), Z (DOS at the energies E_k),
#        E_200 (energies of the 200-well structure), L (its length)
fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
axs[0].plot(k * a / np.pi, E_k)
axs[0].set_xlabel(r"$k a / \pi$")
axs[0].set_ylabel("E (eV)")
axs[1].plot(Z_E, E_k, label="from E(k)")
hist, edges = np.histogram(E_200, bins=30)
axs[1].barh(0.5 * (edges[1:] + edges[:-1]), hist / np.diff(edges) / L,
            height=np.diff(edges), alpha=0.4, label="histogram, N = 200")
axs[1].set_xlabel(r"Z(E) (eV$^{-1}$ nm$^{-1}$)")
axs[1].set_xlim(0, 8 * Z_E.min())
axs[1].legend()
show()

# TODO answer (where does Z(E) diverge and why, comparison with free electron in 1D):
# The density of states is highest at the edges
