"""
Exercise: Carrier statistics and the pn junction

Skeleton code. The material class and the functions in the first cell are
complete and can be used as they are -- read them carefully, you need to
understand what they do. Everything marked with TODO is your job.
Hand-in: this file with ALL TODOs filled in. Running it must produce all
plots. Write the answers to the questions of the exercise sheet as comments into
the "TODO answer" fields. The file is split into cells (# %%),
which you can run one by one in VS Code, Spyder or PyCharm, or you can copy
them into a Jupyter notebook. Functions you still have to write raise a
NotImplementedError -- replace that line by your code.

Units used throughout: SI, i.e. energies in J, densities in m^-3, lengths in m.
For plots, convert to eV (divide by const.e), cm^-3 (multiply by 1e-6) and um.
"""

# %% Imports, material and given functions  (given -- do not change)
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
import scipy.optimize
from scipy.integrate import trapezoid


class BaseMaterial:
    def __init__(self, T=300):
        self.T = T
        # mat.E is the energy grid on which the energy-resolved quantities
        # (DOS(E), f(E), n(E)) are evaluated: 2 eV below E_V to 2 eV above E_C
        self.E = np.linspace(self.Ev / const.e - 2., self.Ec / const.e + 2., 2000) * const.e


class Si(BaseMaterial):
    def __init__(self, N_D, N_A, T=300):
        self.Eg = 1.12 * const.e   # band gap at room temperature (T dependence neglected)
        self.Ec = 0.               # conduction band edge = energy zero
        self.Ev = -self.Eg         # valence band edge
        self.T = T
        self.N_D = N_D             # donor density in m^-3
        self.N_A = N_A             # acceptor density in m^-3

        self.m_l = 0.98 * const.m_e   # longitudinal effective mass, conduction band
        self.m_t = 0.19 * const.m_e   # transverse effective mass, conduction band
        self.m_hh = 0.49 * const.m_e  # heavy-hole effective mass, valence band
        self.m_lh = 0.16 * const.m_e  # light-hole effective mass, valence band

        self.eps_r = 11.7

        N_cb_valleys = 6

        # density-of-states masses
        self.m_dos_n = N_cb_valleys**(2 / 3) * (self.m_l * self.m_t**2)**(1 / 3)
        self.m_dos_p = (self.m_hh**(3 / 2) + self.m_lh**(3 / 2))**(2 / 3)

        # effective densities of states (band weights) N_C, N_V
        self.Nc = 2 / const.h**3 * (2 * np.pi * self.m_dos_n * const.k * T)**(3. / 2)
        self.Nv = 2 / const.h**3 * (2 * np.pi * self.m_dos_p * const.k * T)**(3. / 2)

        # intrinsic carrier density, consistent with N_C, N_V and E_g
        self.n_i = np.sqrt(self.Nc * self.Nv) * np.exp(-self.Eg / (2 * const.k * T))

        super().__init__(T)


def calcDensityOfStates(E_grid, mat, band='CB'):
    """3D density of states Z(E) in J^-1 m^-3 (incl. spin) of the CB or the VB."""
    if band == 'CB':
        E = np.where(E_grid > mat.Ec, E_grid, mat.Ec)
        Z_E = 1 / (2 * np.pi)**3 * (4 * np.pi) / const.hbar**3 * (2 * mat.m_dos_n)**(3 / 2) * np.sqrt(E - mat.Ec)
        assert Z_E[0] == 0
    elif band == 'VB':
        E = np.where(E_grid < mat.Ev, E_grid, mat.Ev)
        Z_E = 1 / (2 * np.pi)**3 * (4 * np.pi) / const.hbar**3 * (2 * mat.m_dos_p)**(3 / 2) * np.sqrt(-(E - mat.Ev))
        assert Z_E[-1] == 0
    return Z_E


def calcProbabilityDensity_FD(E_grid, mat, Ef):
    """Fermi-Dirac occupation probability f(E) for the Fermi energy Ef (exact)."""
    f = 1 / (1 + np.exp((E_grid - Ef) / (const.k * mat.T)))
    return f


# %% Task 1a -- carrier densities from the Fermi energy (Fermi-Dirac)
mySi = Si(0, 0, 300)

# Exemplary use of the given functions:
#   dos = calcDensityOfStates(mySi.E, mySi, 'CB')
#   f = calcProbabilityDensity_FD(mySi.E, mySi, -0.5 * const.e)

# plot Z(E) of both bands and f(E) for a few Fermi energies
# (energy in eV on the vertical axis) to understand the given functions.

dos_cb = calcDensityOfStates(mySi.E, mySi, 'CB')
dos_vb = calcDensityOfStates(mySi.E, mySi, 'VB')

f = calcProbabilityDensity_FD(mySi.E, mySi, -0.5 * const.e)

print(dos_cb.shape)
print(f.shape)

# exit()
# plt.figure(figsize=(6, 4))

EV = 1.602e-19

def to_ev(E_J):
    return E_J / EV

def dos_to_per_ev_cm3(dos):
    dos = np.asarray(dos)
    return dos * EV * 1e-6

fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

axs[0].plot(dos_to_per_ev_cm3(dos_cb), to_ev(mySi.E))
axs[0].plot(dos_to_per_ev_cm3(dos_vb), to_ev(mySi.E))
axs[0].set_ylabel("E (eV)")
axs[0].set_xlabel("Z(E)")

axs[1].plot(f, to_ev(mySi.E))
axs[1].set_ylabel("E (eV)")
axs[1].set_xlabel("f(E)")

plt.show()


def calcElectronDensity_FD(mat, E_Fermi):
    """Electron density n in m^-3 in the CB for the Fermi energy E_Fermi in J."""
    # n = integral Z_CB(E) f(E) dE on the energy grid mat.E

    dos = calcDensityOfStates(mat.E, mat, 'CB')
    f = calcProbabilityDensity_FD(mat.E, mat, E_Fermi)
    # one_min_f = 1 / (1 + np.exp((E_Fermi - mat.E) / (const.k * mat.T)))

    n = np.trapezoid(dos * f, mat.E )
    return n 


def calcHoleDensity_FD(mat, E_Fermi):
    """Hole density p in m^-3 in the VB for the Fermi energy E_Fermi in J."""
    # p = integral Z_VB(E) (1 - f(E)) dE on the energy grid mat.E
    # (see the hint on the sheet on how to compute 1 - f)
    z_v = calcDensityOfStates(mat.E, mat, 'VB')
    one_min_f = 1 / (1 + np.exp((E_Fermi - mat.E) / (const.k * mat.T)))
    p = np.trapezoid(z_v * one_min_f, mat.E )

    return p


print(f"n = {calcElectronDensity_FD(mySi, mySi.Ec - 0.02 * const.e) * 1e-6:.3e} cm^-3")
print(f"p = {calcHoleDensity_FD(mySi, mySi.Ev + 0.02 * const.e) * 1e-6:.3e} cm^-3")
# n = 1.131e+19 cm^-3
# p = 4.080e+18 cm^-3

# %% Task 1b -- n and p over the Fermi energy
# calculate n and p for Fermi energies from E_V - 0.3 eV to E_C + 0.3 eV,
# plot them over E_F in eV (logarithmic y axis in cm^-3) and indicate the
# band gap (e.g. with plt.axvspan).

fermi_energies = np.linspace(mySi.Ev - 0.3 * const.e, mySi.Ec + 0.3 * const.e, 200)
ns = []
ps = []

for fermi in fermi_energies:
    n = calcElectronDensity_FD(mySi, fermi) * 1e-6
    p = calcHoleDensity_FD(mySi, fermi) * 1e-6
    ns.append(n)
    ps.append(p)

middle_band_gap = (mySi.Ec + mySi.Ev) / (2 * const.e)

fig = plt.figure(figsize=(8, 6))
plt.plot(to_ev(np.asarray(fermi_energies)), ns, label="n")
plt.plot(to_ev(np.asarray(fermi_energies)), ps, label="p")
plt.axvline(middle_band_gap, color='black', linestyle='--', label='midgap')
plt.axvspan(mySi.Ev / const.e, mySi.Ec / const.e, color='gray', alpha=0.2, label='band gap')
plt.yscale("log")
plt.legend()
plt.xlabel("E (eV)")
plt.show()

# TODO answer (Where do n and p cross? What is this point called?):
# N and P lines cross when both n and p numbers are the same.
# The point is called: intrinsic Fermi level 


# %% Task 1c -- Maxwell-Boltzmann approximation
def calcProbabilityDensity_MB(E_grid, mat, Ef):
    # Implement the Maxwell-Boltzmann distribution
    f_E = np.exp(-1 * (np.asarray(E_grid) - Ef) / (const.k * mat.T))
    return f_E


def calcElectronDensity_MB(mat, E_Fermi):
    # Use the Maxwell-Boltzmann distribution to calculate the electron density
    n = mat.Nc * np.exp(-1 * (mat.Ec - E_Fermi) / (const.k * mat.T))
    return n


def calcHoleDensity_MB(mat, E_Fermi):
    # The same for the holes
    p = mat.Nv * np.exp(-1 * (E_Fermi - mat.Ev) / (const.k * mat.T))
    return p


# add n and p from the Maxwell-Boltzmann approximation to the plot of Task 1b.
# Find the Fermi energy (relative to E_C, in units of kT) from which the
# MB result is more than 10 % too large, and the corresponding density n.

ns_mb = [] 
ps_mb = []

for fermi_energy in fermi_energies: 
    # f_E =  calcProbabilityDensity_MB(mat.E, mat, fermi_energy)8
    n = calcElectronDensity_MB(mySi, fermi_energy) * 1e-6
    p = calcHoleDensity_MB(mySi, fermi_energy) * 1e-6
    ns_mb.append(n)
    ps_mb.append(p)


ns = np.asarray(ns) # N FD
ns_mb = np.asarray(ns_mb)

mask = ns_mb / ns > 1.1
E_lim_n = to_ev(fermi_energies[mask][0])
kT = mySi.T * const.k / const.e
n_lim_fd = ns[mask][0]
n_lim_mb = ns_mb[mask][0]
diff = n_lim_mb - n_lim_fd

print(f"E_F at 10 % deviation: {E_lim_n:.4f} eV")
print(f"E_C - E_F: {(mySi.Ec / const.e - E_lim_n) / kT:.2f} kT")
print(f"n_FD at 10 % deviation: {(n_lim_fd * 1e-6):.3e} cm^-3")
print(f"n_MB at 10 % deviation: {(n_lim_mb * 1e-6):.3e} cm^-3")
print(f"Difference: {(diff * 1e-6):.3e} cm^-3")
print()

fig = plt.figure(figsize=(8, 6))
plt.plot(to_ev(np.asarray(fermi_energies)), (ns), label="n Fermi-Dirac")
plt.plot(to_ev(np.asarray(fermi_energies)), (ps), label="p Fermi-Dirac")
plt.plot(to_ev(np.asarray(fermi_energies)), (ns_mb), label="n Maxwell-Boltzmann")
plt.plot(to_ev(np.asarray(fermi_energies)), (ps_mb), label="p Maxwell-Boltzmann")
plt.axvline(E_lim_n, color='green', linestyle='--', label='E_F at 10% deviation')
plt.axhline(n_lim_fd, color='darkorange', linestyle=':', label='n at 10% deviation')
plt.axvline(middle_band_gap, color='black', linestyle='--', label='midgap')
plt.axvspan(mySi.Ev / const.e, mySi.Ec / const.e, color='gray', alpha=0.2, label='band gap')
plt.yscale("log")
plt.xlabel("E (eV)")
plt.legend(loc="lower left")
plt.show()


# TODO answer (differences FD vs. MB, where and why, limit in kT and in cm^-3):
# The diffrence of n and p densities between FD and MB appears 
# when fermi energies level are close to valence or conduction bands values. 
# This is due to Pauli exclusion principle which MB does not take into consideration
# The 10% diffrence between FD and MB in electron density appears for Ef = 0.028 eV
# The non-degenerate limit is 1e18 cm^-3, so the approximation is a good
# Same diffrence occurs for holes 


# %% Task 2a -- free carriers from doping
def calcChargeCarrierDensitiesFromDoping(mat):
    # use the formulas from the script to calculate the electron density n
    # and the hole density p from the doping of the material mat
    dN = 0.5 * (mat.N_D - mat.N_A)
    root = np.sqrt(0.25 * (mat.N_D - mat.N_A)**2 + mat.n_i**2)

    if dN >= 0:
        n = dN + root
        p = mat.n_i ** 2 / n
    else: 
        p = -1 * dN + root
        n = mat.n_i ** 2 / p

    return (n, p)


# %% Task 2b -- Fermi energy over doping  (the two functions are given)
def calcEf_Exact(mat):

    n, p = calcChargeCarrierDensitiesFromDoping(mat)

    def fun_e(Ef):
        E_Fermi = Ef * const.e
        return calcElectronDensity_FD(mat, E_Fermi) - n

    def fun_h(Ef):
        E_Fermi = Ef * const.e
        return calcHoleDensity_FD(mat, E_Fermi) - p

    fun = fun_e if mat.N_D > mat.N_A else fun_h

    result_Ef, _ = scipy.optimize.bisect(fun, mat.E[0] / const.e, mat.E[-1] / const.e,
                                              maxiter=1000, full_output=True)
    result_Ef = result_Ef * const.e
    return result_Ef


def calcEf_Approx(mat):
    if mat.N_D > mat.N_A:
        return mat.Ec - const.k * mat.T * np.log(mat.Nc / (mat.N_D - mat.N_A))
    else:
        return mat.Ev + const.k * mat.T * np.log(mat.Nv / (mat.N_A - mat.N_D))


# plot E_F over the doping with both functions, in two plots:
# 1) N_A = 1e12 m^-3 fixed, N_D varied (e.g. 1e14 ... 1e26 m^-3)
# 2) N_D = 1e12 m^-3 fixed, N_A varied
# Logarithmic doping axis; mark E_C, E_V and the intrinsic level E_i.


def ef_curves(vary):
    dopings = np.logspace(14, 26, 100)
    fermi_energies = []

    def get_fermi_energies(si):
        return calcEf_Approx(si), calcEf_Exact(si)

    for d in dopings:
        si = Si(N_D=d, N_A=1e12) if vary == "D" else Si(N_D=1e12, N_A=d)
        approx, exact = get_fermi_energies(si)
        fermi_energies.append((approx, exact))

    return dopings, fermi_energies


fig, axes = plt.subplots(1, 2, figsize=(10, 8))
for ax, vary, doping_label in [(axes[0], "D", "dN_D"), (axes[1], "A", "dN_A")]:
    data = ef_curves(vary)
    dopping = data[0]
    approx_energies, exact_energies = map(np.asarray, zip(*data[1]))

    ax.plot(dopping, to_ev(approx_energies), label="Approx energies (eV)")
    ax.plot(dopping, to_ev(exact_energies), label="Exact energies (eV)")
    ax.axhline(to_ev(mySi.Ec), color="red", linestyle="--", label="E_C")
    ax.axhline(to_ev(mySi.Ev), color="blue", linestyle="--", label="E_V")
    E_i = (mySi.Ec + mySi.Ev) / 2 + (const.k * mySi.T / 2) * np.log(mySi.Nv / mySi.Nc)
    ax.axhline(to_ev(E_i), color="green", linestyle=":", label="$E_i$")

    ax.set_xscale("log")
    ax.set_ylabel("Energy (eV)")
    ax.set_xlabel(doping_label)
    ax.legend()

plt.show()


# TODO answer (approximation(s) behind calcEf_Approx, where and why it fails):
# The approximation method uses Maxwell-Boltzmann approximation for calculating n and p densities. 
# The method assums the ionisation of dopants and that the minority carriers can be neglected
# The exact energies asymptotically tends to intrinsic energy level of the material as the doping N_D, N_A decreases


# %% Task 3 -- built-in voltage
# TODO: calculate the built-in voltage V_builtIn with calcEf_Exact()
#       1) for the example of the lecture notes: N_D = 1e17 cm^-3, N_A = 5e16 cm^-3
#       2) for the doping of a real pn diode (look it up)
#       Remember: Si(N_D, N_A) takes the doping in m^-3.
pSi = Si(N_D=1e10, N_A=5e22, T=300)
nSi = Si(N_D=1e23, N_A=1e10, T=300)

V_builtIn = to_ev(calcEf_Exact(nSi) - calcEf_Exact(pSi))
print(f'Built in voltage: {V_builtIn:.3f} V')

# Ansys PN Junction Diode 
nSi_real = Si(N_D=1e21, N_A=1e10, T=300)
pSi_real = Si(N_D=1e10, N_A=1e23, T=300)

V_real = to_ev(calcEf_Exact(nSi_real) - calcEf_Exact(pSi_real))
print(f"Built-in voltage from Ansys diode: {V_real:.3f} V")

nSi_high = Si(N_D=1e26, N_A=1e10, T=300)
pSi_high = Si(N_D=1e10, N_A=1e26, T=300)

V_high = to_ev(calcEf_Exact(nSi_high) - calcEf_Exact(pSi_high))
print(f"High-doping built-in voltage: {V_high:.3f} V")

# Built in voltage: 0.837 V
# Built-in voltage from Ansys diode: 0.736 V
# High-doping built-in voltage: 1.323 V

# TODO answer (your doping values and source, comparison with real diodes, upper limit of V_bi):
# src: https://optics.ansys.com/hc/en-us/articles/360042273134-pn-junction-diode
# ND = 1e15 cm^-3 and NA = 1e17 cm^-3 from the Ansys Optics
# The calculated voltage of diode is 0.736 V which is in range of the typical silicon diode 0.6 V - 0.7 V   
# For high doping diode the V_bi exceeds the 1.12 V (E_F can move at most from E_V to E_C) E_F therefore must be inside
# the bands - degenerate doping 


# %% Task 4a -- pn junction in equilibrium: Poisson equation
class Poisson:
    def __init__(self, material, device_length):
        num_grid_points = len(material)
        self.numGridPoints = num_grid_points
        self.material = material
        self.device_length = device_length
        self.x = np.linspace(0, device_length, num_grid_points)
        self.dx = self.x[1] - self.x[0]

        # E_F from charge neutrality at the two boundaries
        Ef_boundary_left = calcEf_Exact(self.material[0])
        Ef_boundary_right = calcEf_Exact(self.material[-1])

        # built-in voltage (as in Task 3), sign: phi(L) = phi(0) + V_built_in
        V_built_in = to_ev(Ef_boundary_right - Ef_boundary_left)

        # one Fermi level in the whole device: we take E_F of the left boundary
        # (What changes if another definition is used here?)
        self.Ef = Ef_boundary_left

        # this gives the boundary conditions for the electrostatic potential
        self.V_boundary_left = 0.
        self.V_boundary_right = V_built_in

        # constant of the discretised Poisson equation (all constants combined)
        eps_r = self.returnMaterialParameterAsArray('eps_r')
        self.poissonConst = self.dx**2 / (const.epsilon_0 * eps_r) * const.elementary_charge

    def returnMaterialParameterAsArray(self, attributeName):
        return np.array([getattr(mat, attributeName) for mat in self.material])

    def solve(self):
        initialGuess = self.initialGuess()
        solution = scipy.optimize.root(self.poissonEquation, initialGuess[1:-1], method='lm')
        if solution['success'] == False:
            raise Exception('Poisson equation solver failed. Did not converge.')
        V_without_boundary = solution['x']
        self.V = np.hstack([self.V_boundary_left, V_without_boundary, self.V_boundary_right])
        return self.V

    def initialGuess(self):
        V = np.zeros(self.numGridPoints)
        V[:self.numGridPoints // 2] = self.V_boundary_left
        V[self.numGridPoints // 2:] = self.V_boundary_right
        return V

    def poissonEquation(self, V_without_boundary):
        # equations for the inner grid points
        # Poisson equation  eps d^2V/dx^2 + q (N_D - N_A - n + p) = 0
        V = np.hstack([self.V_boundary_left, V_without_boundary, self.V_boundary_right])

        n = self.calcElectronDensity_Array(V_without_boundary * const.e + self.Ef)
        p = self.calcHoleDensity_Array(V_without_boundary * const.e + self.Ef)

        N_D = self.returnMaterialParameterAsArray('N_D')
        N_A = self.returnMaterialParameterAsArray('N_A')

        # TODO: residual of the discretised Poisson equation at the inner grid
        #       points (a vector of length N - 2), see the sheet
        source_term = self.poissonConst[1:-1] * (N_D[1: -1] - N_A[1:-1] - n + p)
        residual = V[2:] - 2 * V[1:-1] + V[:-2] + source_term

        return residual

    def calcElectronDensity_Array(self, poisson_potential):
        # poisson_potential = E_F + e V(x) at the inner grid points
        n = np.zeros_like(poisson_potential)
        for idx, v in enumerate(poisson_potential):
            n[idx] = calcElectronDensity_FD(self.material[idx + 1], v)
        return n

    def calcHoleDensity_Array(self, poisson_potential):
        p = np.zeros_like(poisson_potential)
        for idx, v in enumerate(poisson_potential):
            p[idx] = calcHoleDensity_FD(self.material[idx + 1], v)
        return p

    def plot(self):
        if not hasattr(self, 'V'):
            raise Exception('Plot failed. Electrostatic potential self.V does not exist.')

        fig, (ax_doping, ax_densities, ax_charge, ax_potentials, ax_energy) = \
            plt.subplots(5, 1, sharex=True, figsize=(6, 9))
        x_um = self.x * 1e6

        N_D = self.returnMaterialParameterAsArray('N_D')
        N_A = self.returnMaterialParameterAsArray('N_A')
        ax_doping.semilogy(x_um, N_D * 1e-6, label=r'$N_D$')
        ax_doping.semilogy(x_um, N_A * 1e-6, label=r'$N_A$')
        ax_doping.set_ylabel('doping (cm$^{-3}$)')
        ax_doping.legend()

        n = np.array([calcElectronDensity_FD(m, v * const.e + self.Ef) for m, v in zip(self.material, self.V)])
        p = np.array([calcHoleDensity_FD(m, v * const.e + self.Ef) for m, v in zip(self.material, self.V)])
        ax_densities.semilogy(x_um, n * 1e-6, label=r'$n$')
        ax_densities.semilogy(x_um, p * 1e-6, label=r'$p$')
        ax_densities.set_ylabel('n, p (cm$^{-3}$)')
        ax_densities.legend()

        ax_charge.plot(x_um, (N_D - N_A - n + p) * 1e-6)
        ax_charge.set_ylabel(r'$N_D-N_A-n+p$ (cm$^{-3}$)')

        ax_potentials.plot(x_um, self.V)
        ax_potentials.set_ylabel('el. potential (V)')

        Ec = self.returnMaterialParameterAsArray('Ec')
        Ev = self.returnMaterialParameterAsArray('Ev')
        ax_energy.plot(x_um, Ec / const.e - self.V, label=r'$E_C$')
        ax_energy.plot(x_um, Ev / const.e - self.V, label=r'$E_V$')
        ax_energy.hlines(self.Ef / const.e, 0, x_um[-1], linestyle='--', color='black', label=r'$E_F$')
        ax_energy.set_ylabel('energy (eV)')
        ax_energy.set_xlabel('x ($\\mu$m)')
        ax_energy.legend(loc='lower right', fontsize=8)

        fig.tight_layout()
        return fig


# %% Task 4b -- simulate the pn junction of the lecture notes  (given, takes ~20 s)
N = 200
L = 3e-6

nSi = Si(N_D=1e23, N_A=1e10, T=300)   # N_D = 1e17 cm^-3
pSi = Si(N_D=1e10, N_A=5e22, T=300)   # N_A = 5e16 cm^-3

# every grid point can contain another material: left half n-type, right half p-type
materialStack = [nSi] * (N // 2) + [pSi] * (N // 2)

pn = Poisson(materialStack, L)

t0 = time.time()
pn.solve()
print(f'solved in {time.time() - t0:.1f} s')
fig = pn.plot()
plt.show()

# TODO answer (potential drop vs. Task 3, width of the space-charge zone on both
#              sides and why, total charge, what changes with another choice of E_F):
#
# The potential drops is above 0.8 V which is similar to Task 3 
# The depletion region n-side is about 0.1 um and 0.2 um for p-side. It is wider on the p-side because it is less doped.
# Total charge on the n side equals the charge on the p side - the charge gets killed - 0 charge overalll.
# Choosing another E_F only shifts potential and bands. The potential drop, widths and charge stay the same