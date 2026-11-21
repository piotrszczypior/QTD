"""
Exercise: Drift and diffusion -- fleas on dogs

Skeleton code. The imports and the plot helper in the first cell are complete
and can be used as they are. Everything marked with TODO is your job.
Hand-in: this file with ALL TODOs filled in. Running it must produce all
plots. Write the answers to the questions of the exercise sheet as comments into
the "TODO answer" fields. The file is split into cells (# %%),
which you can run one by one in VS Code, Spyder or PyCharm, or you can copy
them into a Jupyter notebook. Code you still have to write raises a
NotImplementedError -- replace that line by your code.

Picture of the lecture: the fleas are the charge carriers, the dogs are the
positions (states) the carriers can occupy. Units: SI (s, m, m/s).
"""

# %% Imports and given helper  (given -- do not change)
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig, save_name):
    fig.savefig(PLOTS_DIR / f"{save_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

# use this generator for all random numbers, e.g.
# randomGenerator.choice([0, 1], p=[0.9, 0.1])  ->  0 with probability 0.9, 1 with probability 0.1
randomGenerator = np.random.default_rng()

SNAPSHOT_COLORS = ['C0', 'C1', 'C2', 'C3', 'C4']


def plotSnapshots(x, t, fleaNumber, ax=None, linestyle='-', label=True):
    """Plot the number of fleas on each dog over the dog position x (in nm) at
    the times T/32, T/16, T/8, T/4 and T. fleaNumber has the shape (len(t), len(x))."""
    if ax is None:
        ax = plt.gca()
    for idx, color in zip([len(t) // 32, len(t) // 16, len(t) // 8, len(t) // 4, -1], SNAPSHOT_COLORS):
        ax.plot(x * 1e9, fleaNumber[idx], linestyle=linestyle, color=color,
                label=f't = {t[idx] * 1e12:.2f} ps' if label else None)
    ax.set_xlabel('dog position x (nm)')
    ax.set_ylabel('number of fleas on dog')
    return ax


# %% Task 1a -- two dogs: every flea individually
numFlea = 1000

tau = 0.1e-9         # 0.1 ps, mean time between two jumps of a flea
T = tau * 10            # total duration of the simulation
dt = tau / 10           # time step
t = np.arange(0, T, dt)  # array of times

jumpingProbability = dt / tau  # probability that a flea jumps to the other dog within dt
# jumpingProbability = 0.15
# print(jumpingProbability)
numFleaLeft = np.zeros(len(t), dtype=np.int64)   # number of fleas on the left dog for each time step
numFleaRight = np.zeros(len(t), dtype=np.int64)  # and on the right dog
numFleaLeft[0] = numFlea

# 1. Loop over all time steps iT = 1 ... len(t)-1.
# 2. Loop over all fleas on the left dog at time step iT-1. For each flea, decide with
#    fleaWillJump = randomGenerator.choice([0, 1], p=[1 - jumpingProbability, jumpingProbability])
# whether it stays (0) or jumps to the right dog (1).
# 3. Put the flea on the correct dog at the new time step iT.
# 4. Repeat 2. and 3. for the fleas on the right dog.


for it in range(1, len(t)):
    jump_left = 0
    jump_right = 0

    for flea in range(numFleaLeft[it - 1]):
        fleaWillJump = randomGenerator.choice([0, 1], p=[1 - jumpingProbability, jumpingProbability])
        jump_right =  jump_right + fleaWillJump
        
    for flea in range(numFleaRight[it - 1]):
        fleaWillJump = randomGenerator.choice([0, 1], p=[1 - jumpingProbability, jumpingProbability])
        jump_left =  jump_left + fleaWillJump

    numFleaRight[it] = numFleaRight[it - 1] - jump_left + jump_right
    numFleaLeft[it] = numFleaLeft[it - 1] + jump_left - jump_right


if 1:
    fig = plt.figure()
    plt.plot(t, numFleaLeft, linestyle=':', color='r', label='left dog')
    plt.plot(t, numFleaRight, linestyle=':', color='b', label='right dog')
    plt.plot(t, numFlea * (0.5 + 0.5 * np.exp(-2 * t / tau)), '--', color='lightcoral', label='analytical left')
    plt.plot(t, numFlea * (0.5 - 0.5 * np.exp(-2 * t / tau)), '--', color='dodgerblue', label='analytical right')
    plt.xlabel('time (s)')
    plt.ylabel('number of fleas on dog')
    plt.legend()
    fig.tight_layout()
    save_fig(fig, "lab_04_exercise_1a")

# TODO answer (behaviour for different numFlea and tau):
# When the number of fleas is low, the individual simulation deviates more from the analytical solution.
# With more fleas, these fluctuations become relatively smaller, and the simulation approaches the analytical solution
# Increasing tau makes the redistribution slower in time but the number of simulation steps needed to reach equilibrium 
# remains approximately the same because T = tau * 10 also scales with tau
# When increasing the jumping probability the equilibrium is reach much faster


# %% Task 1b -- two dogs: statistical average
numFlea = 50

tau = 0.1e10
T = tau * 10
dt = tau / 10
t = np.arange(0, T, dt)

jumpingProbability = dt / tau

numFleaLeft = np.zeros(len(t), dtype=np.int64)
numFleaRight = np.zeros(len(t), dtype=np.int64)
numFleaLeft[0] = numFlea

fleaDensityLeft = np.zeros(len(t))   # average number of fleas on the left dog
fleaDensityRight = np.zeros(len(t))  # and on the right dog
fleaDensityLeft[0] = numFlea

# 1. Copy your loop from Task 1a (individual fleas).
# 2. Add the statistical average in the same time loop: on average,
#          jumpingProbability * fleaDensityLeft fleas jump to the right and
#          jumpingProbability * fleaDensityRight fleas jump to the left.
# 3. Use this to calculate fleaDensityLeft and fleaDensityRight at the new
#    time step. Do not loop over the fleas here.

for it in range(1, len(t)):
    jump_left = 0
    jump_right = 0

    for flea in range(numFleaLeft[it - 1]):
        fleaWillJump = randomGenerator.choice([0, 1], p=[1 - jumpingProbability, jumpingProbability])
        jump_right =  jump_right + fleaWillJump
        
    for flea in range(numFleaRight[it - 1]):
        fleaWillJump = randomGenerator.choice([0, 1], p=[1 - jumpingProbability, jumpingProbability])
        jump_left =  jump_left + fleaWillJump

    numFleaRight[it] = numFleaRight[it - 1] - jump_left + jump_right
    numFleaLeft[it] = numFleaLeft[it - 1] + jump_left - jump_right


for it in range(1, len(t)):
    jumping_right = jumpingProbability * fleaDensityLeft[it - 1]
    jumping_left = jumpingProbability * fleaDensityRight[it - 1]
    fleaDensityRight[it] = fleaDensityRight[it - 1] + jumping_right - jumping_left
    fleaDensityLeft[it] = fleaDensityLeft[it - 1] - jumping_right + jumping_left


if 1:
    fig = plt.figure()
    # plt.plot(t, numFleaLeft, linestyle=':', color='r', label='left dog (individual)')
    # plt.plot(t, numFleaRight, linestyle=':', color='b', label='right dog (individual)')
    plt.plot(t, fleaDensityLeft, linestyle='-', color='r', label='left dog (average)')
    plt.plot(t, fleaDensityRight, linestyle='-', color='b', label='right dog (average)')
    plt.plot(t, numFlea * (0.5 + 0.5 * np.exp(-2 * t / tau)), '--', color='lightcoral', label='analytical left')
    plt.plot(t, numFlea * (0.5 - 0.5 * np.exp(-2 * t / tau)), '--', color='dodgerblue', label='analytical right')
    plt.xlabel('time (s)')
    plt.ylabel('number of fleas on dog')
    plt.legend()
    fig.tight_layout()
    save_fig(fig, "lab_04_exercise_1b")

# TODO answer (comparison of the two approaches for different numFlea and tau):
# The number of fleas does not depend on the behavior of the model. 
# It approximates the analytical sollution similarly based on the jumping probability. 
# Similarly when increasing the jumping probabily the equilibrium is reach faster 



# %% Task 2a -- many dogs: diffusion and recombination (statistical average)
def simulateFleaDensity(numFlea, t, x, jumpingProbability, Pl, Pr, Precombination):
    """Average number of fleas fleaDensity[iT, iX] on dog iX at time step iT.
    At t = 0 all fleas sit on the central dog; fleas reaching the outer dogs are lost."""
    fleaDensity = np.zeros((len(t), len(x)))


    #      1. Starting condition: all numFlea fleas on the central dog (index len(x) // 2).
    #       2. Loop over the time steps (as in Task 1b) and over the inner dogs iX = 1 ... len(x)-2.
    #          The new number of fleas on dog iX is made of the fleas that stay,
    #          (1 - jumpingProbability) * fleaDensity[iT-1, iX], the fleas arriving from the left,
    #          Pr * jumpingProbability * fleaDensity[iT-1, iX-1], and the fleas arriving from the
    #          right, Pl * jumpingProbability * fleaDensity[iT-1, iX+1].
    #       3. Boundary conditions in every time step: fleaDensity[iT, 0] = fleaDensity[iT, -1] = 0.
    #       4. Test that it works with Precombination = 0.
    #       5. Include the recombination: in every time step the fraction Precombination
    #          of the fleas is lost.
    fleaDensity[0, len(x) // 2] = numFlea

    for it in range(1, len(t)):
        for ix in range(1, len(x) - 2):
            recombination = 1 - Precombination
            stay = (1 - jumpingProbability) * fleaDensity[it - 1, ix]
            right = Pr * jumpingProbability * fleaDensity[it - 1, ix - 1]
            left = Pl  * jumpingProbability * fleaDensity[it - 1, ix + 1]
            fleaDensity[it, ix] =  recombination * (stay + right + left)
        
        fleaDensity[it, 0] = fleaDensity[it, -1] = 0

    return fleaDensity


numFlea = 1000

tau = 0.1e-12
T = tau * 1000
dt = tau / 10
t = np.arange(0, T, dt)

jumpingProbability = dt / tau

v_th = 2e5            # thermal velocity in m/s
dx = v_th * dt        # distance between two dogs
x = np.arange(-dx * 40, dx * 40, dx)

Pl = Pr = 0.5         # Pl + Pr = 1 always!

tau_recombination = 1e-11
Precombination = dt / tau_recombination

fleaDensity_noRecombination = simulateFleaDensity(numFlea, t, x, jumpingProbability, Pl, Pr, 0.)
fleaDensity_field = simulateFleaDensity(numFlea, t, x, jumpingProbability, Pl, Pr, Precombination)

if 1:
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 4))
    plotSnapshots(x, t, fleaDensity_noRecombination, ax=ax1)
    ax1.set_title('without recombination')
    ax1.legend()
    plotSnapshots(x, t, fleaDensity_field, ax=ax2)
    ax2.set_title(f'with recombination, tau_rec = {tau_recombination:.0e} s')
    ax2.legend()
    fig.tight_layout()
    save_fig(fig, "lab_04_exercise_2a")


# %% Task 2b -- drift: the Shockley-Haynes experiment
#   simulate with an applied field, i.e. asymmetric jumping probabilities
#       (e.g. Pr = 0.6, Pl = 0.4), and plot it like in Task 2a. Then vary the input
#       parameters (numFlea, tau_recombination, T, ...) to answer the questions.
fleaDensity_field = simulateFleaDensity(numFlea, t, x, jumpingProbability, Pl=0.6, Pr=0.4, Precombination=0)
fleaDensity_field_recomb = simulateFleaDensity(numFlea, t, x, jumpingProbability, Pl=0.6, Pr=0.4, Precombination=Precombination)


if 1:
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 4))
    plotSnapshots(x, t, fleaDensity_field, ax=ax1)
    ax1.set_title('without recombination')
    ax1.legend()
    plotSnapshots(x, t, fleaDensity_field_recomb, ax=ax2)
    ax2.set_title(f'with recombination, tau_rec = {tau_recombination:.0e} s')
    ax2.legend()
    fig.tight_layout()
    save_fig(fig, "lab_04_exercise_2b")


# TODO answer (minimal number of fleas? tau_recombination in relation to which other
#              lifetime? what happens for a very large T?):
#
# 1. What is a minimum number of fleas needed for reasonable results?
# There is no resonable minimum for a statistical average model. One flea is sufficient 
#
# 2. A longer recombination lifetime means less recombination: which other typical lifetime does tau rec have to be compared with?
# A longer recombination lifetime means fewer fleas are lost and it can be compared with the time carriers neeed to cross the sample 
#
# 3. What happens if the simulation runs for a very long time (large T)
# Without recombination the distribution spreads - gets lower and wider and eventually when fleas reach outher dogs they disappear
# With recombinaton the fleas disappear due to recombination and the boundary condtions and eventually all feals disappear. 

# %% Task 3 -- many dogs: every flea individually  (takes about 15 s)
def simulateFleaCount(numFlea, t, x, jumpingProbability, Pl, Pr, Precombination):
    """Number of fleas fleaCount[iT, iX] on dog iX at time step iT, every flea
    simulated individually. Same starting and boundary conditions as in Task 2a."""
    #   1. Similar structure to Task 1a. Implement the starting condition.
    #       2. In addition to Task 1a you need a loop over the dogs: loop over all inner
    #          dogs, and for each dog loop over its fleas.
    #       3. For each flea, use
    #              moveFlea = randomGenerator.choice([-1, 0, 1], p=[Pl * jumpingProbability,
    #                                                1 - jumpingProbability, Pr * jumpingProbability])
    #          which tells if the flea moves to the left (-1), stays (0) or moves to the right (1).
    #       4. Boundary conditions as in Task 2a.
    #       5. Try without recombination first, then include the recombination with another
    #          randomGenerator.choice([0, 1], ...).
    fleaCount = np.zeros((len(t), len(x)), dtype=np.int64)
    fleaCount[0, len(x) // 2] = numFlea

    for it in range(1, len(t)):
        for ix in range(1, len(x) - 2):

            for _ in range(fleaCount[it - 1, ix]):
                jump = randomGenerator.choice([-1, 0, 1], p=[Pl*jumpingProbability, 1-jumpingProbability, Pr*jumpingProbability])

                isRecombine = randomGenerator.choice([0, 1], p=[1 - Precombination, Precombination])
                if isRecombine:
                    continue

                fleaCount[it, ix + jump] += 1

        fleaCount[it, 0] = fleaCount[it, -1] = 0

    return fleaCount


numFlea = 1000

tau = 0.1e-12
T = tau * 100
dt = tau / 10
t = np.arange(0, T, dt)

jumpingProbability = dt / tau

v_th = 2e5
dx = v_th * dt
x = np.arange(-dx * 40, dx * 40, dx)

Pl = Pr = 0.5

tau_recombination = 1e-11
Precombination = dt / tau_recombination

if 1:
    t0 = time.time()
    fleaCount = simulateFleaCount(numFlea, t, x, jumpingProbability, Pl, Pr, 0)
    fleaCount_precombination = simulateFleaCount(numFlea, t, x, jumpingProbability, Pl, Pr, Precombination)

    print(f'Task 3: individual fleas simulated in {time.time() - t0:.1f} s')
    fleaDensity_noRecomb = simulateFleaDensity(numFlea, t, x, jumpingProbability, Pl, Pr, 0)
    fleaDensity_field = simulateFleaDensity(numFlea, t, x, jumpingProbability, Pl, Pr, Precombination)

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 4))
    plotSnapshots(x, t, fleaCount, linestyle='--', label=False, ax=ax1)
    plotSnapshots(x, t, fleaDensity_noRecomb, ax=ax1)
    ax1.plot([], [], 'k--', label='individual')
    ax1.plot([], [], 'k-', label='average')
    ax1.set_title('without recombination')
    ax1.legend(fontsize=8)

    plotSnapshots(x, t, fleaCount_precombination, linestyle=':', label=False, ax=ax2)
    plotSnapshots(x, t, fleaDensity_field, ax=ax2)
    ax2.plot([], [], 'k:', label='individual')
    ax2.plot([], [], 'k-', label='average')
    ax2.set_title(f'with recombination, tau_rec = {tau_recombination:.0e} s')
    ax2.legend(fontsize=8)

    fig.suptitle('Individual fleas vs. statistical average')
    fig.tight_layout()
    save_fig(fig, "lab_04_exercise_3")


# %% Task 4 -- constant injection from the left: short and long sample
def simulateInjection(numFlea, t, x, jumpingProbability, Pl, Pr, Precombination):
    """Like simulateFleaDensity, but with a constant injection from the left:
    dog 0 always carries numFlea fleas, the last dog absorbs all fleas."""
    fleaDensity = np.zeros((len(t), len(x)))
    #    copy your code from Task 2a and change only the starting and boundary conditions:
    #       at t = 0, fleaDensity[0, 0] = numFlea and all other dogs are empty;
    #       in every time step, fleaDensity[iT, 0] = numFlea and fleaDensity[iT, -1] = 0.
    #       Hint: with T = 10000 tau there are 1e5 time steps. A loop over the dogs takes some
    #       seconds; replacing it by NumPy slices (fleaDensity[iT-1, :-2], ...) is much faster.
    fleaDensity[0, len(x) // 2] = numFlea

    recombination = 1 - Precombination
    p_stay = 1 - jumpingProbability
    p_right = Pr * jumpingProbability
    p_left = Pl * jumpingProbability
    
    for it in range(1, len(t)):
        stay = p_stay * fleaDensity[it - 1, 1:-1]
        right = p_right * fleaDensity[it - 1, :-2]
        left = p_left * fleaDensity[it - 1, 2:]

        fleaDensity[it, 1:-1] = recombination * (stay + right + left)
        fleaDensity[it, 0] = numFlea
        fleaDensity[it, -1] = 0
   
    return fleaDensity


numFlea = 1000

tau = 0.1e-12
T = tau * 5000
dt = tau / 10
t = np.arange(0, T, dt)

jumpingProbability = dt / tau

v_th = 2e5
dx = v_th * dt
x = np.arange(-dx * 40, dx * 40, dx)

Pl = Pr = 0.5

tau_recombination = 5e-10
Precombination = dt / tau_recombination

if 1:
    fleaDensity_field = simulateInjection(numFlea, t, x, jumpingProbability, Pl, Pr, Precombination)
    fig = plt.figure()
    plotSnapshots(x, t, fleaDensity_field)
    plt.title('Constant injection snapshots')
    plt.legend()
    fig.tight_layout()
    save_fig(fig, "lab_04_exercise_4_1")


if 1:
    fig, ax = plt.subplots(figsize=(10, 5))

    for T_factor in [1000, 1500, 2000, 5000, 7500, 10000, 20000]:
        T = tau * T_factor
        t = np.arange(0, T, dt)

        fleaDensity_field = simulateInjection(numFlea, t, x, jumpingProbability, Pl, Pr, Precombination)
        # plotSnapshots(x, t, fleaDensity_field, ax=ax)

        ax.plot(x * 1e9, fleaDensity_field[-1, :], label=rf'T = {T_factor} $\tau$')

    ax.set_xlabel('dog position x (nm)')
    ax.set_ylabel('number of fleas on dog')
    ax.set_title('Steady-state convergence for different durations T')
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "lab_04_exercise_4_2")

T = tau * 5000
t = np.arange(0, T, dt)

if 1:
    tau_recomb_values = [2e-9, 5e-12]

    D = v_th**2 * tau / 2
    W = x[-1] - x[0]
    x_rel = x - x[0]

    for idx, tau_recomb in enumerate(tau_recomb_values):
        precombination = dt / tau_recomb

        fleaDensity_field = simulateInjection(numFlea, t, x, jumpingProbability, Pl, Pr, precombination)

        Lp = np.sqrt(D * tau_recomb) / 10
        analytical = numFlea * np.sinh((W - x_rel) / Lp) / np.sinh(W / Lp)

        print(f"tau_recombination = {tau_recomb:.1e}")
        print(f"Diffusion length: {Lp * 1e9:.2f} nm")
        print(f"Sample length: {W * 1e9:.2f} nm")
        print(f"W / Lp: {W / Lp:.2f}")
        print()

        # difference = np.max(np.abs(fleaDensity_field[-1] - fleaDensity_field[len(t) // 2]))
        # print(difference)
        # ax.plot(x * 1e9, fleaDensity_field[-1, :], label=f'tau_rec = {tau_recomb:.1e} s')

        fig_n = plt.figure()
        plt.plot(x * 1e9, fleaDensity_field[-1], label='Numerical')
        plt.plot(x * 1e9, analytical, '--', label='Analytical')
        plt.xlabel('dog position x (nm)')
        plt.ylabel('number of fleas on dog')
        plt.title(rf'Steady-state density ($\tau_{{\mathrm{{rec}}}}$ = {tau_recomb:.1e} s)')
        plt.legend()
        fig_n.tight_layout()
        save_fig(fig_n, f"lab_04_exercise_4_{2 * idx + 3}")

        current = jumpingProbability * Pr * fleaDensity_field[-1, :-1] - Pl * fleaDensity_field[-1, 1:]
        fig_j = plt.figure()
        plt.plot(x[:-1] * 1e9, current, label='Steady-state current')
        plt.xlabel('dog position x (nm)')
        plt.ylabel('current (a.u.)')
        plt.title(rf'Steady-state current ($\tau_{{\mathrm{{rec}}}}$ = {tau_recomb:.1e} s)')
        plt.legend()
        fig_j.tight_layout()
        save_fig(fig_j, f"lab_04_exercise_4_{2 * idx + 4}")
  

# plt.figure()
# fleaDensity_field = simulateInjection(numFlea, t, x, jumpingProbability, Pl, Pr, Precombination)
# plotSnapshots(x, t, fleaDensity_field)

# plt.legend()
# plt.show()

#   1. Increase T until the solution hardly changes any more (steady state).
#       2. Find one set of parameters for a short sample and one for a long sample and
#          simulate both.
#       3. Implement the analytical steady-state solution
#              p(x) = p(0) sinh((W - x) / Lp) / sinh(W / Lp)
#          with x measured from the left end of the sample, W the length of the sample, and
#              D  = v_th**2 * tau / 2
#              Lp = np.sqrt(D * tau_recombination) / 10
#          (the factor 10 comes from dt = tau / 10, see the sheet) and add it to the plots.
#       4. Calculate the current along the sample in steady state (arbitrary units) and plot
#          it over x in a new figure.

# TODO answer (time to reach the steady state, your parameters for the short and the long
#              sample, comparison with the analytical solution, where does the current go?):
#
# Time to reach the steady state: tau * 5000 
# 
# Short sample 
# tau_recombination = 2.0e-09
# Diffusion length: 200.00 nm
# Sample length: 158.00 nm
# W / Lp: 0.79

# Long samole
# tau_recombination = 5.0e-12
# Diffusion length: 10.00 nm
# Sample length: 158.00 nm
# W / Lp: 15.80

# Short sample -linear concentration and constant current
# Long sample - exponential concentration and exponential decay current  

# where does the current go?
# Decreases due to recombination
