import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

ncores = [
    256,
    512,
    1024,
    2048,
]
nhrs = [
    1500,
    2e03,
    3e03,
    5e03,
]


# def lnr(x, a, b):
#     return a * x + b
# def poly2(x, a, b, c):
#     return a * x**2 + b * x + c
# def inv(x, a, b):
#     return a / x + b
def log(x, a, b):
    return a * np.log10(x) + b


# overhead = [x - nhrs[0] for x in nhrs]
eff = [771.0 / x for x in nhrs]  # Myr/CPUh
eff = [x / eff[0] for x in eff]  # Myr/CPUh

time_to_sol = [(x / y) * 3600.0 for x, y in zip(nhrs, ncores)]  # s
time_to_sol = [time_to_sol[0] / x for x in time_to_sol]  # s

fig, ax = plt.subplots(1, 1, figsize=(8, 8), layout="constrained")

ax.scatter(ncores, time_to_sol, label="Measurements")

# xlim = (128, 4096)
xlim = ax.get_xlim()
ax.set_xlim(xlim)


lin_args, pcov = curve_fit(log, ncores, time_to_sol)

print(lin_args)

perfect_cores = np.linspace(256, xlim[1], 20)
perfect_line = [(x / perfect_cores[0]) for x in perfect_cores]
# perfect_line = [
#     time_to_sol[0] for x in perfect_cores
# ]  # perfect scaling: increasing the number of tasks has
# no overhead - aka 2* as many cores goes 2x as fast


ax.plot(
    perfect_cores,
    log(perfect_cores, *lin_args),
    ls=":",
    c="tab:blue",
    # label=f"y = {lin_args[0]:.3e}xx + {lin_args[1]:.3e}x + {lin_args[2]:.3f}",
    label=f"y = {lin_args[0]:.2f} log(x) + {lin_args[1]:.2f}",
    # label=f"y = {lin_args[0]:.3e}/x + {lin_args[1]:.1f}",
)

ax.plot(perfect_cores, perfect_line, "--k", label="Perfect scaling")

ax.set_xlabel("Number of cores", size="larger")
ax.set_ylabel("Speedup", size="larger")

# ax.set_yscale("log")

ax.legend(framealpha=0.0)

fig.suptitle("Strong scaling test: Time to reach z=8", size="x-large")

fig.savefig("strong_scaling.png")
