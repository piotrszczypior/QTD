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
import time
import numpy as np
import matplotlib.pyplot as plt

# use this generator for all random numbers, e.g.
#   randomGenerator.choice([0, 1], p=[0.9, 0.1])  ->  0 with probability 0.9, 1 with probability 0.1
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

tau = 0.1e-12           # 0.1 ps, mean time between two jumps of a flea
T = tau * 10            # total duration of the simulation
dt = tau / 10           # time step
t = np.arange(0, T, dt)  # array of times

jumpingProbability = dt / tau  # probability that a flea jumps to the other dog within dt

numFleaLeft = np.zeros(len(t), dtype=np.int64)   # number of fleas on the left dog for each time step
numFleaRight = np.zeros(len(t), dtype=np.int64)  # and on the right dog
numFleaLeft[0] = numFlea

# TODO: 1. Loop over all time steps iT = 1 ... len(t)-1.
#       2. Loop over all fleas on the left dog at time step iT-1. For each flea, decide with
#              fleaWillJump = randomGenerator.choice([0, 1], p=[1 - jumpingProbability, jumpingProbability])
#          whether it stays (0) or jumps to the right dog (1).
#       3. Put the flea on the correct dog at the new time step iT.
#       4. Repeat 2. and 3. for the fleas on the right dog.

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


plt.figure()
plt.plot(t, numFleaLeft, linestyle=':', color='r', label='left dog')
plt.plot(t, numFleaRight, linestyle=':', color='b', label='right dog')
plt.plot(t, numFlea * (0.5 + 0.5 * np.exp(-2 * t / tau)), '--', color='lightcoral', label='analytical right')
plt.plot(t, numFlea * (0.5 - 0.5 * np.exp(-2 * t / tau)), '--', color='dodgerblue', label='analytical left')
plt.xlabel('time (s)')
plt.ylabel('number of fleas on dog')
plt.legend()
plt.show()

# TODO answer (behaviour for different numFlea and tau):
#


# %% Task 1b -- two dogs: statistical average
numFlea = 100

tau = 0.1e-12
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

# TODO: 1. Copy your loop from Task 1a (individual fleas).
#       2. Add the statistical average in the same time loop: on average,
#          jumpingProbability * fleaDensityLeft fleas jump to the right and
#          jumpingProbability * fleaDensityRight fleas jump to the left.
#       3. Use this to calculate fleaDensityLeft and fleaDensityRight at the new
#          time step. Do not loop over the fleas here.

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


plt.figure()
plt.plot(t, numFleaLeft, linestyle=':', color='r', label='left dog (individual)')
plt.plot(t, numFleaRight, linestyle=':', color='b', label='right dog (individual)')
plt.plot(t, fleaDensityLeft, linestyle='-', color='r', label='left dog (average)')
plt.plot(t, fleaDensityRight, linestyle='-', color='b', label='right dog (average)')
plt.plot(t, numFlea * (0.5 + 0.5 * np.exp(-2 * t / tau)), '--', color='lightcoral', label='analytical')
plt.plot(t, numFlea * (0.5 - 0.5 * np.exp(-2 * t / tau)), '--', color='dodgerblue')
plt.xlabel('time (s)')
plt.ylabel('number of fleas on dog')
plt.legend()
plt.show()

# TODO answer (comparison of the two approaches for different numFlea and tau):
#



# %% Task 2a -- many dogs: diffusion and recombination (statistical average)
def simulateFleaDensity(numFlea, t, x, jumpingProbability, Pl, Pr, Precombination):
    """Average number of fleas fleaDensity[iT, iX] on dog iX at time step iT.
    At t = 0 all fleas sit on the central dog; fleas reaching the outer dogs are lost."""
    fleaDensity = np.zeros((len(t), len(x)))


    # TODO: 1. Starting condition: all numFlea fleas on the central dog (index len(x) // 2).
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
            # fleaDensity[] FIXME: boundary condition

    return fleaDensity


numFlea = 1000


tau = 0.1e-12
T = tau * 100
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

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 4))
plotSnapshots(x, t, fleaDensity_noRecombination, ax=ax1)
ax1.set_title('without recombination')
ax1.legend()
plotSnapshots(x, t, fleaDensity_field, ax=ax2)
ax2.set_title(f'with recombination, tau_rec = {tau_recombination:.0e} s')
plt.show()


# %% Task 2b -- drift: the Shockley-Haynes experiment
# TODO: simulate with an applied field, i.e. asymmetric jumping probabilities
#       (e.g. Pr = 0.6, Pl = 0.4), and plot it like in Task 2a. Then vary the input
#       parameters (numFlea, tau_recombination, T, ...) to answer the questions.
fleaDensity_field = simulateFleaDensity(numFlea, t, x, jumpingProbability, Pl=0.6, Pr=0.4, Precombination=0)
fleaDensity_field = simulateFleaDensity(numFlea, t, x, jumpingProbability, Pl=0.6, Pr=0.4, Precombination=Precombination)

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 4))
plotSnapshots(x, t, fleaDensity_field, ax=ax1)
ax1.set_title('without recombination')
ax1.legend()
plotSnapshots(x, t, fleaDensity_field, ax=ax2)
ax2.set_title(f'with recombination, tau_rec = {tau_recombination:.0e} s')
plt.show()


# TODO answer (minimal number of fleas? tau_recombination in relation to which other
#              lifetime? what happens for a very large T?):
#


# %% Task 3 -- many dogs: every flea individually  (takes about 15 s)
def simulateFleaCount(numFlea, t, x, jumpingProbability, Pl, Pr, Precombination):
    """Number of fleas fleaCount[iT, iX] on dog iX at time step iT, every flea
    simulated individually. Same starting and boundary conditions as in Task 2a."""
    # TODO: 1. Similar structure to Task 1a. Implement the starting condition.
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

t0 = time.time()
fleaCount = simulateFleaCount(numFlea, t, x, jumpingProbability, Pl, Pr, 0)
fleaCount_precombination = simulateFleaCount(numFlea, t, x, jumpingProbability, Pl, Pr, Precombination)

print(f'Task 3: individual fleas simulated in {time.time() - t0:.1f} s')
fleaDensity_field = simulateFleaDensity(numFlea, t, x, jumpingProbability, Pl, Pr, Precombination)


fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 4))
plotSnapshots(x, t, fleaCount, linestyle='--', label=False, ax=ax1)
plotSnapshots(x, t, fleaDensity_field, ax=ax1)

plotSnapshots(x, t, fleaCount_precombination, linestyle=':', label=False, ax=ax2)
plotSnapshots(x, t, fleaDensity_field, ax=ax2)

plt.title('individual fleas (dashed) and statistical average (solid)')
plt.legend()
plt.show()


# %% Task 4 -- constant injection from the left: short and long sample
def simulateInjection(numFlea, t, x, jumpingProbability, Pl, Pr, Precombination):
    """Like simulateFleaDensity, but with a constant injection from the left:
    dog 0 always carries numFlea fleas, the last dog absorbs all fleas."""
    fleaDensity = np.zeros((len(t), len(x)))
    # TODO: copy your code from Task 2a and change only the starting and boundary conditions:
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


# a good set of parameters:
numFlea = 1000

tau = 0.1e-12
T = tau * 10000
dt = tau / 10
t = np.arange(0, T, dt)

jumpingProbability = dt / tau

v_th = 2e5
dx = v_th * dt
x = np.arange(-dx * 40, dx * 40, dx)

Pl = Pr = 0.5

tau_recombination = 5e-10
Precombination = dt / tau_recombination

fleaDensity_field = simulateInjection(numFlea, t, x, jumpingProbability, Pl, Pr, Precombination)

plt.figure()
plotSnapshots(x, t, fleaDensity_field)
plt.legend()
plt.show()

# TODO: 1. Increase T until the solution hardly changes any more (steady state).
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
