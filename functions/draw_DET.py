"""
Description: 
Author: Xiongjun Guan
Date: 2023-09-20 20:00:24
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2023-09-20 20:43:19

Copyright (C) 2023 by Xiongjun Guan, Tsinghua University. All rights reserved.
"""

import os.path as osp
import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
import math


def compute_roc(score_mat, num_steps=100, gms_only=False):
    """compute ROC curve.

    Parameters:
        score_mat: a dict contains {"genuine": ..., "impostor": ...}
        gms_only: generally for VeriFinger, only genunie scores are required
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


def draw_det_on_ax(ax, data, label, color, linewidth=2, linestyle="-", xmin=None):
    """_summary_

    Args:
        ax (_type_): _description_
        data (_type_): a dict contains {"gms": genuine match scores, "ims": imposter match scores}
        label (_type_): line label
        color (_type_): line color
        linewidth (int, optional): line width. Defaults to 2.
        linestyle (str, optional): line style. Defaults to "-".
        xmin (_type_, optional): Interpolate the position of point with minimum x. Defaults to None.
    """
    gms = np.array(data["gms"]).reshape((-1,))
    ims = np.array(data["ims"]).reshape((-1,))
    rate_arr = compute_roc({"genuine": gms, "impostor": ims}, num_steps=500)
    if xmin is not None:
        ys = rate_arr[1]
        xs = rate_arr[0]
        ys = ys[xs > 0]
        xs = xs[xs > 0]
        ymin = np.interp(-np.log(xmin), -np.log(xs), ys)
        ys = ys[xs > xmin]
        xs = xs[xs > xmin]

        xs = np.hstack((xs, xmin))
        ys = np.hstack((ys, ymin))

        rate_arr = np.vstack((xs, ys))
    ax.plot(
        rate_arr[0],
        1 - rate_arr[1],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
    )


if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(111)

    data_dir = f"./test_data/det/verifinger/"
    draw_det_on_ax(
        ax,
        scio.loadmat(osp.join(data_dir, "original.mat")),
        "Without Rectification :   VeriFinger",
        "r",
        linestyle="-",
    )
    draw_det_on_ax(
        ax,
        scio.loadmat(osp.join(data_dir, "Gu.mat")),
        "Gu et al. :   VeriFinger",
        "orange",
        linestyle="-",
    )
    draw_det_on_ax(
        ax,
        scio.loadmat(osp.join(data_dir, "Dabouei_timg.mat")),
        "Dabouei et al. :   VeriFinger",
        "g",
        linestyle="-",
    )
    draw_det_on_ax(
        ax,
        scio.loadmat(osp.join(data_dir, "DDRNet_timg.mat")),
        "Proposed Method :   VeriFinger",
        "b",
    )
    draw_det_on_ax(
        ax,
        scio.loadmat(osp.join(data_dir, "DDRNet_withOri_timg.mat")),
        "Proposed Method +O :   VeriFinger",
        "c",
        linestyle="-",
    )

    data_dir = f"./test_data/det/mcc/"
    draw_det_on_ax(
        ax,
        scio.loadmat(osp.join(data_dir, "original.mat")),
        "MCC",
        "r",
        linestyle="--",
        xmin=math.pow(10, -2.98),
    )
    draw_det_on_ax(
        ax,
        scio.loadmat(osp.join(data_dir, "Gu.mat")),
        "MCC",
        "orange",
        linestyle="--",
        xmin=math.pow(10, -2.98),
    )
    draw_det_on_ax(
        ax,
        scio.loadmat(osp.join(data_dir, "Dabouei_timg.mat")),
        "MCC",
        "g",
        linestyle="--",
        xmin=math.pow(10, -2.98),
    )
    draw_det_on_ax(
        ax,
        scio.loadmat(osp.join(data_dir, "DDRNet_timg.mat")),
        "MCC",
        "b",
        linestyle="--",
        xmin=math.pow(10, -2.98),
    )
    draw_det_on_ax(
        ax,
        scio.loadmat(osp.join(data_dir, "DDRNet_withOri_timg.mat")),
        "MCC",
        "c",
        linestyle="--",
        xmin=math.pow(10, -2.98),
    )

    ax.tick_params(axis="both", labelsize=12)
    ax.minorticks_on()

    ax.set_ylim(0, 0.4)

    ax.set_xscale("log")
    ax.set_xlim([1e-3, 1])
    ax.set_xlabel("FMR", fontsize=12)
    ax.set_ylabel("FNMR", fontsize=12)
    ax.grid(which="major", axis="x", linestyle="--", linewidth=0.5)
    ax.grid(which="both", axis="y", linestyle="--", linewidth=0.5)

    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.set_title("DET Curves on FVC2004_DB1_A Hard Subset", fontsize=12)

    ax.legend(ncol=2, fontsize=10, markerfirst=False)

    fig.tight_layout()

    fig.savefig(f"./results/det.png", bbox_inches="tight", dpi=500)

    plt.close(fig)
