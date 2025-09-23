import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

ncores = [
    128,
    208,
    512,
]
nhrs = [
    3e2,
    6e2,
    2.5e3,
]


def lnr(x, a, b):
    return a * x + b


# def poly2(x, a, b, c):
#     return a * x**2 + b * x + c
# def inv(x, a, b):
#     return a / x + b


# overhead = [x - nhrs[0] for x in nhrs]
eff = [771.0 / x for x in nhrs]  # Myr/CPUh
eff = [x / eff[0] for x in eff]  # Myr/CPUh

time_to_sol = [(x / y) * 3600.0 for x, y in zip(nhrs, ncores)]  # s
# time_
time_to_sol = [time_to_sol[0] / x for x in time_to_sol]  # s

print(time_to_sol)

fig, ax = plt.subplots(1, 1, figsize=(8, 8), layout="constrained")

ax.scatter(ncores, time_to_sol, label="Measurements")

# xlim = (128, 4096)
xlim = ax.get_xlim()
ax.set_xlim(xlim)


lin_args, pcov = curve_fit(lnr, ncores, time_to_sol)

perfect_cores = np.linspace(128, xlim[1], 20)

perfect_line = [(x / perfect_cores[0]) for x in perfect_cores]  # speed up

# print(list(zip(perfect_cores, perfect_line)))

# perfect_line = [time_to_sol[0] for x in perfect_cores]
# perfect_line = [
#     time_to_sol[0] for x in perfect_cores
# ]  # perfect scaling: increasing the number of tasks has
# no overhead - aka 2* as many cores goes 2x as fast


ax.plot(
    perfect_cores,
    lnr(perfect_cores, *lin_args),
    ls=":",
    c="tab:blue",
    # label=f"y = {lin_args[0]:.3e}xx + {lin_args[1]:.3e}x + {lin_args[2]:.3f}",
    label=f"y = {lin_args[0]:.3f}x + {lin_args[1]:.1f}",
)

ax.plot(
    perfect_cores,
    perfect_line,
    "--k",
    label=f"Perfect scaling, slope: {time_to_sol[0] / perfect_cores[0]:.3f}",
)

ax.set_xlabel("Number of cores", size="larger")
ax.set_ylabel(f"Normalized speedup", size="larger")

# ax.set_yscale("log")

# ax.set_ylim(ax.get_ylim()[0], 2e4)

ax.legend(framealpha=0.0)

fig.suptitle("Weak scaling test: Time to reach z=8", size="x-large")

fig.savefig("weak_scaling.png")
