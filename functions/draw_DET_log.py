"""
Description: 
Author: Xiongjun Guan
Date: 2023-09-20 19:38:42
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2023-09-21 10:33:05

Copyright (C) 2023 by Xiongjun Guan, Tsinghua University. All rights reserved.
"""

import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import math


def compute_roc(score_mat, num_steps=100, gms_only=False):
    """compute ROC curve.

    Parameters:
        score_mat: a dict contains {"genuine": ..., "impostor": ...}
        gms_only: generally for VeriFinger, only genunie scores are required, s = -12 x log_10(FMR)
    Returns:
        rate_arr: [false-positive-rate; true-positive-rate]
    """
    genuine_scores = np.array(score_mat["genuine"]).astype(np.float32).flatten()

    if gms_only:
        num_steps = min(num_steps, len(genuine_scores))
        rate_arr = np.zeros([2, num_steps])

        rate_arr[0] = np.linspace(-8, 0, num_steps)
        threshs = -rate_arr[0] * 12
        for ii, c_th in enumerate(threshs):
            rate_arr[1, ii] = (genuine_scores >= c_th).sum() * 1.0 / len(genuine_scores)
        rate_arr[0] = 10 ** rate_arr[0]
    else:
        impostor_scores = np.array(score_mat["impostor"]).astype(np.float32).flatten()
        num_steps = min(num_steps, len(genuine_scores) + len(impostor_scores))
        rate_arr = np.zeros([2, num_steps])

        threshs = np.concatenate(
            (
                np.linspace(
                    min(genuine_scores.min(), impostor_scores.min()),
                    max(genuine_scores.max(), impostor_scores.max()),
                    num_steps // 2,
                ),
                np.random.choice(
                    np.concatenate((genuine_scores, impostor_scores)),
                    num_steps - num_steps // 2,
                    replace=False,
                ),
            )
        )
        threshs = np.sort(threshs)
        for ii, c_th in enumerate(threshs):
            rate_arr[0, ii] = (
                (impostor_scores >= c_th).sum() * 1.0 / len(impostor_scores)
            )
            rate_arr[1, ii] = (genuine_scores >= c_th).sum() * 1.0 / len(genuine_scores)

    return rate_arr


def draw_det_on_ax_log(
    ax,
    gms,
    ims,
    label,
    color,
    linewidth=2,
    linestyle="-",
    xmin=1e-4,
    ymin=1e-3,
    ymax=1,
    need_eer=True,
    need_grid=True,
):
    """_summary_

    Args:
        ax (_type_): _description_
        gms (_type_): genuine match scores.
        ims (_type_): imposter match scores.
        label (_type_): _description_
        color (_type_): _description_
        linewidth (int, optional): _description_. Defaults to 2.
        linestyle (str, optional): _description_. Defaults to "-".
        xmin (_type_, optional): _description_. Defaults to 1e-4.
        ymin (_type_, optional): _description_. Defaults to 1e-3.
        ymax (int, optional): _description_. Defaults to 1.
        need_eer (bool, optional): _description_. Defaults to True.
        need_grid (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    rate_arr = compute_roc({"genuine": gms, "impostor": ims}, num_steps=30000)
    ys = 1 - rate_arr[1]
    xs = rate_arr[0]

    idx = np.argmin(np.abs(ys - xs))
    eer = (ys[idx] + xs[idx]) / 2

    idx = np.argmin(np.abs(xs - 1e-3))
    FMR1000 = (ys[idx - 1] + ys[idx + 1] + ys[idx]) / 3

    idx = np.argmin(np.abs(xs - 1e-2))
    FMR100 = (ys[idx - 1] + ys[idx + 1] + ys[idx]) / 3

    min_val = np.min(xs[xs > 0])
    idxs = np.where(xs == min_val)[0]
    ZeroFMR = np.mean(ys[idxs])

    ax.plot(
        rate_arr[0],
        1 - rate_arr[1],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
    )

    # EER line
    if need_eer:
        points = np.array([0, 1])
        ax.plot(points, points, linewidth=linewidth, linestyle="-", color="k")

        if ymax == 1:
            ax.text(
                np.power(10, -1.45),
                np.power(10, -1.35),
                "EER",
                color="darkred",
                rotation=45,
            )
        elif ymax == 1e-1:
            ax.text(
                np.power(10, -2.45),
                np.power(10, -2.35),
                "EER",
                color="darkred",
                rotation=45,
            )

    # draw wide grid lines
    if need_grid:
        for i in range(-6, 2):
            y = np.array([math.pow(10, i), math.pow(10, i)])
            x = np.array([math.pow(10, -5), math.pow(10, 0)])
            ax.plot(x, y, linewidth=linewidth, linestyle="-", color="k")
        for j in range(-6, 2):
            y = np.array([math.pow(10, -4), math.pow(10, 0)])
            x = np.array([math.pow(10, j), math.pow(10, j)])
            ax.plot(x, y, linewidth=linewidth, linestyle="-", color="k")

        if ymin <= 1e-4:
            ax.text(
                np.power(10, -3.95),
                np.power(10, -0.75 + np.log10(ymax)),
                "FMR10000",
                color="darkred",
                rotation=90,
            )
        ax.text(
            np.power(10, -2.95),
            np.power(10, -0.75 + np.log10(ymax)),
            "FMR1000",
            color="darkred",
            rotation=90,
        )
        ax.text(
            np.power(10, -1.95),
            np.power(10, -0.75 + np.log10(ymax)),
            "FMR100",
            color="darkred",
            rotation=90,
        )

    # draw minor line
    ax.minorticks_on()
    ax.grid(which="minor", axis="both", linestyle="--", linewidth=0.5)

    ax.set_xscale("log")
    ax.set_xlim([xmin, 1])
    ax.set_yscale("log")
    ax.set_ylim([ymin, ymax])
    ax.set_xlabel("FMR")
    ax.set_ylabel("FNMR")
    ax.grid(which="major", axis="both", linestyle="--", linewidth=0.5)

    return eer, FMR100, FMR1000, ZeroFMR


if __name__ == "__main__":
    fig = plt.figure()
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
        "Proposed",
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

        eer, FMR100, FMR1000, ZeroFMR = draw_det_on_ax_log(
            ax,
            gms,
            ims,
            label,
            color,
            linestyle=linestyle,
            xmin=1e-3,
            ymin=1e-3,
            need_eer=True,
            need_grid=True,
        )
        print(
            "{}: FMR1000={:.2f}, ZeroFMR=={:.2f}, EER={:.2f}".format(
                method, (1 - FMR1000) * 100, ZeroFMR * 100, eer * 100
            )
        )

    # ax.set_title("Image Correlation Score Curves on FVC2004 DB1_A")
    ax.legend(ncol=1, fontsize=8, loc="upper right", framealpha=1)
    fig.tight_layout()
    fig.savefig(f"./results/det_log.png", bbox_inches="tight", dpi=500)
    plt.close(fig)
