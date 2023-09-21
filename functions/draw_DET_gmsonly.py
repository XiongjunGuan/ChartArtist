"""
Description: 
Author: Xiongjun Guan
Date: 2023-09-20 19:24:21
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2023-09-20 20:03:38

Copyright (C) 2023 by Xiongjun Guan, Tsinghua University. All rights reserved.
"""

import os.path as osp
import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np


def DrawDET_gmsOnly(ax, gms, linewidth, color, linestyle="-", label=None):
    """Calculate DET curves from genuine matching score based on VeriFinger mapping relationship.
        VeriFinger score = -12 x log_10 (FMR)

    Args:
        ax (_type_): axis of figure
        gms (_type_): scores with shape like (n,)
        linewidth (_type_): line width
        color (_type_): line color
        linestyle (str, optional): line sty;e. Defaults to "-".
        label (_type_, optional): label of line. Defaults to None.
    """
    FAR = np.arange(-8, 0.5, 0.2)

    threshold = -FAR * 12  # mapping relationship of VeriFinger

    TAR = np.zeros((len(FAR), 1))
    for i in range(len(FAR)):
        TAR[i] = (sum(gms > threshold[i])) / len(gms)

    FMR = np.power(10, FAR)
    FNMR = (1 - TAR).reshape((-1,))

    ax.plot(
        FMR, FNMR, linewidth=linewidth, color=color, linestyle=linestyle, label=label
    )

    ax.set_xscale("log")
    ax.set_xlabel("FMR", fontsize=12)
    ax.set_ylabel("FNMR", fontsize=12)
    ax.minorticks_on()

    ax.grid(which="minor", axis="both", linestyle="--", linewidth=0.5)
    ax.set_title("DET curve using Verifinger")


def draw_det_on_ax_gms_only(
    ax, data, label, color, linewidth=2, linestyle="-", mask=None
):
    """Use subsets to calculate DET curves.

    Args:
        ax (_type_): axis of figure
        data (_type_): "gms" genuine match scores
        label (_type_): line label
        color (_type_): line color
        linewidth (int, optional): line width. Defaults to 2.
        linestyle (str, optional): line style. Defaults to "-".
        mask (_type_, optional): mask of subset, select if > 0. Defaults to None.
    """
    gms = np.array(data["gms"]).reshape((-1,))
    if mask is not None:
        gms = gms[mask > 0]
    DrawDET_gmsOnly(
        ax, gms, color=color, label=label, linewidth=linewidth, linestyle=linestyle
    )
    return


if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(111)

    data_dir = f"./test_data/det_gmsonly/"
    draw_det_on_ax_gms_only(
        ax,
        scio.loadmat(osp.join(data_dir, "original.mat")),
        "Without Rectification :   VeriFinger",
        "r",
        linestyle="-",
    )
    draw_det_on_ax_gms_only(
        ax,
        scio.loadmat(osp.join(data_dir, "Gu.mat")),
        "Gu et al. :   VeriFinger",
        "orange",
        linestyle="-",
    )
    draw_det_on_ax_gms_only(
        ax,
        scio.loadmat(osp.join(data_dir, "Dabouei_timg.mat")),
        "Dabouei et al. :   VeriFinger",
        "g",
        linestyle="-",
    )
    draw_det_on_ax_gms_only(
        ax,
        scio.loadmat(osp.join(data_dir, "DDRNet_timg.mat")),
        "Proposed Method :   VeriFinger",
        "b",
    )
    draw_det_on_ax_gms_only(
        ax,
        scio.loadmat(osp.join(data_dir, "DDRNet_withOri_timg.mat")),
        "Proposed Method +O :   VeriFinger",
        "c",
        linestyle="-",
    )

    ax.tick_params(axis="both", labelsize=12)
    ax.minorticks_on()
    ax.set_xlim(1e-8, 1)

    ax.set_ylim(0, 0.14)

    ax.set_xlabel("FMR", fontsize=12)
    ax.set_ylabel("Mapped FNMR", fontsize=12)
    ax.grid(which="both", axis="both", linestyle="--", linewidth=0.5)
    ax.set_xticks(np.power(10, np.arange(-8, 1, 1).astype(np.double)))

    ax.set_title("DET Curves on FVC2004_DB1_A (Mapped)")
    ax.legend(ncol=1, fontsize=10, markerfirst=False)

    fig.tight_layout()
    fig.savefig(f"./results/det_gmsonly.png", bbox_inches="tight", dpi=500)
    plt.close(fig)
