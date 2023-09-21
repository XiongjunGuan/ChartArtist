"""
Description: 
Author: Xiongjun Guan
Date: 2023-09-21 10:32:41
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2023-09-21 10:38:38

Copyright (C) 2023 by Xiongjun Guan, Tsinghua University. All rights reserved.
"""

import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if __name__ == "__main__":
    fig = plt.figure(figsize=(7, 3))
    ax = fig.add_subplot(111)

    true_dir = f"./test_data/det_log/gms/"
    false_dir = f"./test_data/det_log/ims/"

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
        "-",
        "-",
        "-",
        "-",
        "-",
    ]

    label_lst = [
        "        TPS Based",
        "      Phase Based",
        "      DRN (local)",
        "     DRN (global)",
        "         Proposed",
    ]

    for i in range(len(method_lst)):
        method = method_lst[i]
        label = label_lst[i]
        color = color_lst[i]
        linestyle = linestyle_lst[i]

        data = np.load(osp.join(true_dir, method + ".npy"), allow_pickle=True).item()
        gms = data["score_lst"]
        data = np.load(osp.join(false_dir, method + ".npy"), allow_pickle=True).item()
        ims = data["score_lst"]

        sns.kdeplot(gms, label=label + " : Genuine Match", color=color, linestyle="-")

    for i in range(len(method_lst)):
        method = method_lst[i]
        label = label_lst[i]
        color = color_lst[i]
        linestyle = linestyle_lst[i]

        data = np.load(osp.join(true_dir, method + ".npy"), allow_pickle=True).item()
        gms = data["score_lst"]
        data = np.load(osp.join(false_dir, method + ".npy"), allow_pickle=True).item()
        ims = data["score_lst"]
        sns.kdeplot(ims, label="Imposter Match", color=color, linestyle="--")

    ax.set_xlim([-0.1, 0.9])
    ax.set_yticks([])
    ax.set_xlabel("Image Correlation Score")
    ax.set_ylabel("Distribution")
    # ax.set_title("Probability Density Estimation Curves on FVC2004 DB1_A")
    ax.legend(ncol=2, fontsize=8, loc="upper right", markerfirst=False)
    fig.tight_layout()
    fig.savefig(f"./results/kde.png", bbox_inches="tight", dpi=500)
    plt.close(fig)
