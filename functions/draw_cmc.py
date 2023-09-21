"""
Description: 
Author: Xiongjun Guan
Date: 2023-09-21 10:37:56
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2023-09-21 10:40:26

Copyright (C) 2023 by Xiongjun Guan, Tsinghua University. All rights reserved.
"""

import os.path as osp
import matplotlib.pyplot as plt
import numpy as np


def compute_cmc(scores, sdim=0, max_rank=30):
    N = scores.shape[0]
    ranks = np.zeros((max_rank,))
    for i in range(N):
        if sdim == 0:
            arrs = scores[i, :]
        elif sdim == 1:
            arrs = scores[:, i]
        s = scores[i, i]
        idx = np.sum(arrs > s)
        if idx < (max_rank):
            ranks[idx:] += 1
    rate_arr = 100 * ranks / N
    return rate_arr


def draw_cmc(ax, rank_arr, color="k", linestyle="-", linewidth=1.5, label=""):
    x = np.arange(len(rank_arr)) + 1
    ax.plot(
        x, rank_arr, color=color, linestyle=linestyle, linewidth=linewidth, label=label
    )


if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(111)

    true_dir = f"./test_data/cmc/gms/"
    false_dir = f"./test_data/cmc/ims/"

    method_lst = [
        "end_tps",
        "end_phase",
        "end_DRN_local",
        "end_DRN_global",
        "end_BBRNet_L4",
    ]
    color_lst = [
        "silver",
        "limegreen",
        "orange",
        "plum",
        "dodgerblue",
    ]
    linestyle_lst = [
        "--",
        "--",
        "-",
        "-",
        "-",
    ]

    label_lst = [
        "TPS Based",
        "Phase Based",
        "DRN (local)",
        "DRN (global)",
        "Proposed method",
    ]

    for i in range(len(method_lst)):
        method = method_lst[i]
        label = label_lst[i]
        color = color_lst[i]
        linestyle = linestyle_lst[i]

        data = np.load(osp.join(true_dir, method + ".npy"), allow_pickle=True).item()
        gms = data["score_lst"]
        gftitles = data["ftitle_lst"]
        data = np.load(osp.join(false_dir, method + ".npy"), allow_pickle=True).item()
        ims = data["score_lst"]
        iftitles = data["ftitle_lst"]

        data_tps = np.load(osp.join(true_dir, "end_tps.npy"), allow_pickle=True).item()
        gms_tps = data_tps["score_lst"]
        gftitles_tps = data_tps["ftitle_lst"]

        scores = -np.ones((258, 258))
        for i in range(len(gms)):
            gtmp = gftitles[i].split("_")
            gidx = int(gtmp[0]) - 1

            if gftitles[i] in gftitles_tps:
                idx_tps = gftitles_tps.index(gftitles[i])
                s_tps = gms_tps[idx_tps]
            scores[gidx, gidx] = max(gms[i], s_tps)

        for i in range(len(ims)):
            itmp = iftitles[i].split("_")
            iidx0 = int(itmp[0]) - 1
            iidx1 = int(itmp[2]) - 1
            scores[iidx0, iidx1] = ims[i]

        rank_arr = compute_cmc(scores, max_rank=20)
        draw_cmc(ax, rank_arr, color=color, linestyle=linestyle, label=label)

    xticks = []
    xlabels = []
    for i in [0, 5, 10, 15, 20]:
        xticks.append(i)
        xlabels.append(str(i))
    ax.set_xticks(ticks=xticks, labels=xlabels)
    ax.set_xlim(0, 21)

    yticks = []
    ylabels = []
    for i in [0, 15, 30, 45, 60, 75]:
        yticks.append(i)
        ylabels.append(str(i))
    ax.set_yticks(ticks=yticks, labels=ylabels)
    ax.set_ylim(0, 75)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Identification Rate(%)")
    ax.grid(which="major", axis="y", linestyle="--", linewidth=0.5)

    # ax.set_title("Image Correlator on NIST27")
    ax.legend(ncol=1, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"./results/cmc.png", bbox_inches="tight", dpi=500)
    plt.close(fig)
