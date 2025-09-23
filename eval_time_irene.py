from math import log
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def get_timing_from_log(fpath):
    # works by finding running time= and a= pairs

    l_run = []
    l_as = []
    found = False

    # print(fpath)

    with open(fpath, "r") as src:
        try:
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
        except UnicodeDecodeError:
            print("weird non uni-code chars... assuming file ended")
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
    first_as = []
    first_ts = []

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

        first_as.append(l_as[0])
        first_ts.append(l_run[0])

        tot_runs.extend(l_run)
        tot_as.extend(l_as)

    return tot_runs, tot_as, first_ts, first_as


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
    log_files = np.asarray([f for f in os.listdir(sim_path) if f.endswith(".log")])

    run_nb = [f.split("_")[1].split(".")[0] for f in log_files]

    order = np.argsort(run_nb)

    sim_cpus = []
    sim_cores = []
    first_sim_cpus = []
    first_sim_cores = []

    last_time = 0
    for log_file in log_files[order]:

        l_run, l_as = get_timing_from_log(os.path.join(sim_path, log_file))

        if len(l_run) == 0:
            continue

        if l_run[0] < last_time:
            l_run = [t + last_time for t in l_run]
        last_time = l_run[-1]

        lmin = min(len(l_run), len(l_as))

        with open(os.path.join(sim_path, log_file), "r") as src:

            try:
                for lnb, line in enumerate(src):

                    if "Working with nproc =" in line:

                        # print(line, line.split(" "), line.lstrip(" ").split(" "))

                        elems = [s for s in line.lstrip(" ").split(" ") if s != ""]

                        if "nthr" not in line:
                            sim_cpus.append(np.full(lmin, int(elems[4])))
                            sim_cores.append(np.full(lmin, 1))
                            first_sim_cpus.append(int(elems[4]))
                            first_sim_cores.append(1)
                        else:
                            sim_cpus.append(np.full(lmin, int(elems[4])))
                            sim_cores.append(np.full(lmin, int(elems[8])))
                            first_sim_cpus.append(int(elems[4]))
                            first_sim_cores.append(int(elems[8]))
            except UnicodeDecodeError:
                print("weird non uni-code chars... assuming file ended")
    return (
        np.concatenate(sim_cpus),
        np.concatenate(sim_cores),
        np.asarray(first_sim_cpus),
        np.asarray(first_sim_cores),
    )


def estimate_time_to_z(zeds, times, tgt_z, ax=None, color=None):

    # polyfit order ?, then find closest time to tgt_z

    nfit = 100  # nb of points to use for fit

    p = np.polyfit(zeds[-nfit:], np.log10(times[-nfit:]), 1)

    tgt_time = 10 ** np.polyval(p, tgt_z)

    if ax != None:

        zbins = np.linspace(np.max(zeds), tgt_z)
        # print(zbins)
        ax.plot(10 ** np.polyval(p, zbins), zbins, "o", color=color)

    return tgt_time


def plot_timing_of_dir_sims(ds, tgt_z=None):

    # either pointing to one sim's dir
    # or pointing to a dir that contains sims

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    # fig=plt.figure()#layout='constrained')
    # gs = GridSpec(2, 2, figure=fig)

    # ax = fig.add_subplot(gs[0,0])
    # ax_time = fig.add_subplot(gs[0,1])
    # tab_ax = fig.add_subplot(gs[1,0])
    # fig_tab=plt.figure()
    # tab_ax=fig_tab.add_subplot(111)

    ax = axs[0]
    tab_ax = axs[1]

    # ax.set_aspect('equal')
    # ax_time.set_aspect('equal')

    # table_rows_heads = []
    # table_cols_head = [
    #     "lvlmax",
    #     "ncores",
    #     "current\n z",
    #     "current\n kCPUh",
    #     f"time to\n z={tgt_z:.2f}, kCPUh",
    #     f"time to\n z={tgt_z:.1f}, days",
    # ]

    labels = []
    handles = []

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

            dirs_in_dir = [""]

        else:

            dirs_in_dir = [f for f in f_in_dir if os.path.isdir(os.path.join(d, f))]

        #        print(dirs_in_dir)

        for dir_in_dir in dirs_in_dir:

            #           print(dir_in_dir)

            if check_if_sim_dir(
                os.path.join(d, dir_in_dir), os.listdir(os.path.join(d, dir_in_dir))
            ):

                runs, aexps, first_runs, first_aexps = chain_log_timings(
                    os.path.join(d, dir_in_dir)
                )

                if len(aexps) == 0:
                    continue

                #                print(dir_in_dir.rstrip("/").split("/"))

                sim_name = os.path.join(d, dir_in_dir).rstrip("/").split("/")[-1]
                if sim_name == "":
                    sim_name = d

                if len(sim_name) > 15:
                    sim_name = sim_name[:15] + "\n" + sim_name[15:]

                # table_rows_heads.append(sim_name)

                # print(dir_in_dir, sim_name)

                sim_procs, sim_threads, first_sim_procs, first_sim_threads = (
                    get_sim_proc_count(os.path.join(d, dir_in_dir))
                )

                # print(len(runs), len(sim_procs))

                sim_res = get_sim_lvlmax(os.path.join(d, dir_in_dir))

                zeds = 1.0 / np.asarray(aexps) - 1
                first_zeds = 1.0 / np.asarray(first_aexps) - 1.0
                # print(np.min(aexps), zeds.max())

                # print(len(sim_procs),len(runs))
                # print(len(first_sim_procs),len(first_runs))

                # print(sim_name, sim_procs)
                runs_cpuh = np.cumsum(
                    np.diff(runs) / (3600 / (sim_procs * sim_threads)[1:])
                )
                first_cpuh = np.cumsum(
                    np.diff(first_runs)
                    / (3600 / (first_sim_procs * first_sim_threads)[1:])
                )

                (l,) = ax.plot(
                    runs_cpuh,
                    # [np.asarray(run) / (3600 / sim_procs) for run in runs],
                    zeds[1:],
                    # aexps,
                )

                labels.append(
                    f"{sim_name:s}, lvlmax={sim_res:d}, ncores={np.max(sim_procs):d}, nthreads={np.max(sim_threads):d}"
                )
                handles.append(l)

                # ax.scatter(first_cpuh,
                #         first_zeds,
                #         marker='o',
                #            facecolors='none',
                #         edgecolors=l.get_color())

                if tgt_z != None:

                    tgt_time = estimate_time_to_z(
                        zeds, runs_cpuh, tgt_z
                    )  # , ax=ax, color=l.get_color()
                    # )

                    # ax_time.plot((((runs_cpuh[:]) / sim_procs / 24.0)),
                    #          zeds,
                    #          color=l.get_color())

                    lvl_maxs.append(str(sim_res))
                    ncores.append(str(np.max(sim_procs * sim_threads)))
                    current_zs.append("%.2f" % zeds[-1])
                    current_cpuhs.append("%.1f" % (runs_cpuh[-1] / 1e3))
                    times_to_tgt.append("%.1f" % ((tgt_time - runs_cpuh[-1]) / 1e3))
                    times_to_tgt_days.append(
                        "%.1f"
                        % (
                            (
                                (tgt_time - runs_cpuh[-1])
                                / (np.max(sim_procs * sim_threads))
                                / 24.0
                            )
                        )
                    )

                    print(f"z={zeds[-1]:.2f}")
                    print(f"Estimated time to z={tgt_z} for {sim_name}: {tgt_time}")
                    print(f"remaining time: {tgt_time - runs_cpuh[-1]} cpu hours")
                    print(
                        f"this is {(tgt_time - runs_cpuh[-1])/(np.max(sim_procs*sim_threads))/24} days"
                    )

                    ax.axvline(tgt_time, color=l.get_color(), linestyle="--")

    ax.set_xlabel("CPU hours")
    ax.set_ylabel("Redshift")

    ax.axvline(750000, color="black", linestyle="--", label="10% total time")

    ax.legend(framealpha=0.0)
    # ax.grid()

    ax.set_ylim(2, 8.0)
    ax.set_xlim(1e3, min(ax.get_xlim()[1], 1e7))
    # ax_time.set_ylim(2, 14)
    # ax.set_xlim(0, 1e6)

    ax.set_xscale("log")

    tab_ax.axis("off")

    tab_ax.legend(labels=labels, handles=handles)

    # data = [
    #     lvl_maxs,
    #     ncores,
    #     current_zs,
    #     current_cpuhs,
    #     times_to_tgt,
    #     times_to_tgt_days,
    # ]
    # table = tab_ax.table(
    #     cellText=np.transpose(data),
    #     colLabels=table_cols_head,
    #     rowLabels=table_rows_heads,
    #     loc="center",
    #     # colWidths=np.full_like(table_cols_head, 0.1),
    # )

    # table.auto_set_font_size(False)
    # table.set_fontsize(10)

    # table.set_fontsize(17)
    # table.scale(1.5, 1.5)

    return fig, ax  # , fig_tab


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
    )

    args = parser.parse_args()

    #    fig, ax, fig_tab = plot_timing_of_dir_sims(args.sim_dir, tgt_z=args.tgt_z)
    fig, ax = plot_timing_of_dir_sims(args.sim_dir, tgt_z=args.tgt_z)

    plt.tight_layout()

    ax.set_ylim(args.tgt_z, 8.0)
    # ax.set_xlim(1e3)

    fig.savefig("timing_plot.png")
    # fig_tab.savefig("tab_timing_plot.png")

    plt.show()


# for s in sim_dirs:
#      ...:     try:
#      ...:         runs,aexps = chain_log_timings(s)
#      ...:         print(len(runs))
#      ...:
#      ...:         ax.plot(np.as
# array(runs)/(3600/128),1./np.asarray(aexps)-1,label=s)
#      ...:         print(s)
#      ...:     except:
#      ...:         IndexError
#      ...:
#      ...:         pass
#      ...:
