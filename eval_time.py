import numpy as np
import os
import argparse
import matplotlib.pyplot as plt


def get_timing_from_log(fpath):
    # works by finding running time= and a= pairs

    l_run = []
    l_as = []
    found = False

    with open(fpath, "r") as src:
        for lnb, line in enumerate(src):

            if "running" in line:
                found = False
                l_run.append(float(line.split(":")[1][:-3]))
            if not found:
                if "a=" in line:
                    idx1 = line.index("a=")
                    idx2 = line.index("mem=")
                    l_as.append(float(line[idx1 + 2 : idx2]))
                    found = True

    # print(fpath, len(l_run), len(l_as))

    # if len(l_run) > 0:
    #     print(np.min(l_run), np.max(l_run), np.min(l_as), np.max(l_as))

    return (l_run, l_as)


def chain_log_timings(sim_path):

    log_files = [f for f in os.listdir(sim_path) if f.endswith(".log")]
    log_numbers = [int(f.split("_")[1].split(".")[0]) for f in log_files]

    # order log files

    log_files = [f for _, f in sorted(zip(log_numbers, log_files))]

    tot_runs = []
    tot_as = []

    # print(log_files)
    last_time = 0
    for log_file in log_files:
        l_run, l_as = get_timing_from_log(os.path.join(sim_path, log_file))
        # print(l_run, l_as)

        if len(l_run) == 0:
            continue

        if len(l_run) > len(l_as):
            l_run = l_run[: len(l_as)]
        elif len(l_run) < len(l_as):
            l_as = l_as[: len(l_run)]

        if l_run[0] < last_time:
            l_run = [t + last_time for t in l_run]
        last_time = l_run[-1]

        # print(np.min(l_run), np.max(l_run), np.min(l_as), np.max(l_as))

        tot_runs.extend(l_run)
        tot_as.extend(l_as)

    return tot_runs, tot_as


def check_if_sim_dir(d, f_in_dir):
    nml_in_dir = np.any([f.endswith(".nml") for f in f_in_dir])
    outputs_in_dir = np.any(
        [
            f.startswith("output_") and os.path.isdir(os.path.join(d, f))
            for f in f_in_dir
        ]
    )

    # print(nml_in_dir, outputs_in_dir)

    is_sim_dir = nml_in_dir * outputs_in_dir
    return is_sim_dir


def get_sim_lvlmax(sim_path):

    # find levelmax= in the .nml file

    nml_files = [f for f in os.listdir(sim_path) if f.endswith(".nml")]

    pick_nml = nml_files[0]

    with open(os.path.join(sim_path, pick_nml), "r") as src:

        for lnb, line in enumerate(src):

            if "levelmax" in line:

                # print(line, line.split(" "), line.lstrip(" ").split(" "))

                return int(line.split("=")[1])


def get_sim_proc_count(sim_path):

    # get a log file
    log_files = [f for f in os.listdir(sim_path) if f.endswith(".log")]

    pick_log = log_files[0]

    with open(os.path.join(sim_path, pick_log), "r") as src:

        for lnb, line in enumerate(src):

            if "Working with nproc =" in line:

                # print(line, line.split(" "), line.lstrip(" ").split(" "))

                split_line = np.asarray(line.lstrip(" ").split(" "))
                split_line = split_line[split_line != ""]
                equals_arg = np.where(split_line == "=")

                # print(split_line, split_line == "=", equals_arg)

                return int(split_line[equals_arg[0][0] + 1])


def estimate_time_to_z(zeds, times, tgt_z, ax=None, color=None):

    # polyfit order ?, then find closest time to tgt_z

    if len(zeds) < 1:
        return None

    p = np.polyfit(zeds[-25:], np.log10(times[-25:]), 1)

    tgt_time = 10 ** np.polyval(p, tgt_z)

    if ax != None:

        zbins = np.linspace(np.max(zeds), tgt_z)
        # print(zbins)
        ax.plot(10 ** np.polyval(p, zbins), zbins, "o", color=color)

    return tgt_time


def plot_timing_of_dir_sims(ds, tgt_z=None):

    # either pointing to one sim's dir
    # or pointing to a dir that contains sims

    fig, axs = plt.subplots(2, 1, figsize=(8, 8))

    ax = axs[0]
    tab_ax = axs[1]

    table_rows_heads = []
    table_cols_head = [
        "lvlmax",
        "ncores",
        "current\n z",
        "current\n kCPUh",
        f"time to\n z={tgt_z:.1f}, kCPUh",
        f"time to\n z={tgt_z:.1f}, days",
    ]

    lvl_maxs = []
    ncores = []
    current_zs = []
    current_cpuhs = []
    times_to_tgt = []
    times_to_tgt_days = []

    for d in ds:

        f_in_dir = os.listdir(d)

        is_sim_dir = check_if_sim_dir(d, f_in_dir)

        if is_sim_dir:

            dirs_in_dir = [d]

        else:

            dirs_in_dir = [f for f in f_in_dir if os.path.isdir(f)]

        for dir_in_dir in dirs_in_dir:

            if check_if_sim_dir(dir_in_dir, os.listdir(dir_in_dir)):
                runs, aexps = chain_log_timings(dir_in_dir)

                sim_name = dir_in_dir.rstrip("/").split("/")[-1]

                table_rows_heads.append(sim_name)

                # print(dir_in_dir, sim_name)

                sim_procs = get_sim_proc_count(dir_in_dir)

                sim_res = get_sim_lvlmax(dir_in_dir)

                zeds = 1.0 / np.asarray(aexps) - 1

                # print(np.min(aexps), zeds.max())

                # print(sim_name, sim_procs)
                runs_cpuh = np.asarray(runs) / (3600 / sim_procs)

                (l,) = ax.plot(
                    runs_cpuh,
                    # [np.asarray(run) / (3600 / sim_procs) for run in runs],
                    zeds,
                    # aexps,
                    label=f"{sim_name:s}, lvlmax={sim_res:d}, ncores={sim_procs:d}",
                )

                if tgt_z != None:

                    tgt_time = estimate_time_to_z(
                        zeds, runs_cpuh, tgt_z
                    )  # , ax=ax, color=l.get_color()
                    # )

                    lvl_maxs.append(str(sim_res))
                    ncores.append(str(sim_procs))
                    current_zs.append("%.2f" % zeds[-1])
                    current_cpuhs.append("%.1f" % (runs_cpuh[-1] / 1e3))
                    times_to_tgt.append("%.1f" % ((tgt_time - runs_cpuh[-1]) / 1e3))
                    times_to_tgt_days.append(
                        "%.1f" % (((tgt_time - runs_cpuh[-1]) / sim_procs / 24.0))
                    )

                    print(f"Current z for {sim_name}: {zeds[-1]}")
                    print(f"Estimated time to z={tgt_z} for {sim_name}: {tgt_time}")
                    print(f"remaining time: {tgt_time - runs_cpuh[-1]} cpu hours")
                    print(f"this is {(tgt_time - runs_cpuh[-1])/sim_procs/24} days")

                    ax.axvline(tgt_time, color=l.get_color(), linestyle="--")

    ax.set_xlabel("CPU hours")
    ax.set_ylabel("Redshift")

    ax.axvline(750000, color="black", linestyle="--", label="10% total time")

    ax.legend(framealpha=0.0)
    ax.grid()

    ax.set_ylim(2, 10)
    # ax.set_xlim(0, 1e6)

    ax.set_xscale("log")

    data = [
        lvl_maxs,
        ncores,
        current_zs,
        current_cpuhs,
        times_to_tgt,
        times_to_tgt_days,
    ]

    tab_ax.axis("off")
    table = tab_ax.table(
        cellText=np.transpose(data),
        colLabels=table_cols_head,
        rowLabels=table_rows_heads,
        loc="center",
        # colWidths=np.full_like(table_cols_head, 0.1),
    )

    table.auto_set_font_size(False)
    table.set_fontsize(15)

    # table.set_fontsize(17)
    # table.scale(1.5, 1.5)

    return fig, ax


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plot timing of simulations")
    parser.add_argument(
        "sim_dir",
        metavar="sim_dir",
        type=str,
        nargs="+",
        help="Directory containing simulations or a single simulation",
    )

    parser.add_argument(
        "--tgt_z",
        metavar="tgt_z",
        type=float,
        help="Target redshift to estimate time for",
        default=2.0,
    )

    args = parser.parse_args()

    fig, ax = plot_timing_of_dir_sims(args.sim_dir, tgt_z=args.tgt_z)

    plt.tight_layout()

    fig.savefig("timing_plot.png")

    plt.show()


# for s in sim_dirs:
#      ...:     try:
#      ...:         runs,aexps = chain_log_timings(s)
#      ...:         print(len(runs))
#      ...:
#      ...:         ax.plot(np.asarray(runs)/(3600/128),1./np.asarray(aexps)-1,label=s)
#      ...:         print(s)
#      ...:     except:
#      ...:         IndexError
#      ...:
#      ...:         pass
#      ...:
