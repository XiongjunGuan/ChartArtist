'''
Description: 
Author: Xiongjun Guan
Date: 2023-09-20 19:24:21
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2023-09-21 10:36:12

Copyright (C) 2023 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp


def draw_bar(
    ax,
    x,
    score,
    bar_width,
    seg_score=0.5,
    color="r",
    label=None,
    edgecolor=None,
    hatch=None,
):
    """Draw a bar chart of cumulative distribution

    Args:
        ax (_type_): axis of figure
        x (_type_): horizontal axis value
        score (_type_): vrtical axis value
        bar_width (_type_): bar width
        seg_score (float, optional): the highest value of the score segment. Defaults to 0.5.
        color (str, optional): bar color. Defaults to 'r'.
        label (_type_, optional): annotation of bar. Defaults to None.
        edgecolor (_type_, optional): color of bar edge. Defaults to None.
        hatch (_type_, optional): model of bar filling. Defaults to None.
    """
    score = np.array(score)
    y = np.sum(score <= seg_score) / len(score)
    ax.bar(
        x,
        y,
        width=bar_width,
        color=color,
        label=label,
        hatch=hatch,
        edgecolor=edgecolor,
    )
    # ax.text(x,y,"%.2f"%y,ha='center', va='bottom', fontsize=6)
    return


def draw_cumsum(ax, score, segs, color="r", linestyle="-", marker=None, label=None):
    """Draw a line of cumulative distribution

    Args:
        ax (_type_): axis of figure
        score (_type_): list of scores
        segs (_type_): the highest value of the score segment
        color (str, optional): line color. Defaults to 'r'.
        linestyle (str, optional): line format. Defaults to '-'.
        marker (_type_, optional): shape of data points. Defaults to None.
        label (_type_, optional): annotation of lines. Defaults to None.
    """
    score = np.array(score)
    ys = []
    for seg in segs:
        ys.append(np.sum(score <= seg))
    ys = np.array(ys) / len(score)
    ax.plot(segs, ys, color, linestyle=linestyle, label=label, marker=marker)
    return


if __name__ == "__main__":
    seg_score1 = 0.55

    method_lst = [
        "end_phase",
        "end_DRN_local",
        "end_DRN_global",
        "end_BBRNet_L4",
    ]
    color_lst = ["silver", "limegreen", "orange", "deepskyblue"]
    linestyle_lst = [
        "--",
        "-",
        "-",
        "-",
    ]
    label_lst = [
        "Phase based",
        "DRN (local)",
        "DRN (global)",
        "Proposed method",
    ]

    bar_width = 2

    delta = 2.4
    xdis_lst = [-1.5 * delta, -0.5 * delta, 0.5 * delta, 1.5 * delta]
    num = len(method_lst)

    basedir = f"./test_data/cumsum/"

    step = 0.01
    segs = np.arange(0, 1 + step, step)

    fig = plt.figure(figsize=(12, 8))
    left, bottom, width, height = 0.08, 0.08, 0.84, 0.84
    ax = fig.add_axes([left, bottom, width, height])

    for i in range(len(method_lst)):
        method = method_lst[i]
        label = label_lst[i]
        color = color_lst[i]
        linestyle = linestyle_lst[i]
        method_path = osp.join(basedir, f"{method}.npy")
        data = np.load(method_path, allow_pickle=True).item()

        subset_score = data["subset_score"]
        subset1_score = data["subset1_score"]
        subset2_score = data["subset2_score"]
        subset3_score = data["subset3_score"]

        draw_cumsum(
            ax,
            subset1_score,
            segs,
            color=color,
            linestyle=linestyle,
            marker="o",
            label=label + " (subset 1)",
        )
        draw_cumsum(
            ax,
            subset2_score,
            segs,
            color=color,
            linestyle=linestyle,
            marker="|",
            label=label + " (subset 2)",
        )
        draw_cumsum(
            ax,
            subset3_score,
            segs,
            color=color,
            linestyle=linestyle,
            marker="x",
            label=label + " (subset 3)",
        )
        draw_cumsum(
            ax,
            subset_score,
            segs,
            color=color,
            linestyle=linestyle,
            marker="^",
            label=label + " (whole set)",
        )

    ax.plot([seg_score1, seg_score1], [-1, 2], color="dimgrey", linestyle="--")

    ax.set_xlabel("Correlation scores", size=12)
    ax.set_ylabel("Rate", size=12)
    ax.legend(loc="lower right", fontsize=10)

    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    sets = np.arange(0, 1.1, 0.1)
    setslabel = []
    for s in sets:
        setslabel.append("{:.1f}".format(s))
    ax.set_xticks(sets)
    ax.set_yticks(sets)
    ax.set_xticklabels(setslabel, fontsize=12)
    ax.set_yticklabels(setslabel, fontsize=12)

    ax.set_title("Cumulative distribution curves of genuine matches on FVC2004_DB1_A")

    left, bottom, width, height = 0.13, 0.55, 0.3, 0.33
    ax2 = fig.add_axes([left, bottom, width, height])
    for i in range(len(method_lst)):
        method = method_lst[i]
        label = label_lst[i]
        xdis = xdis_lst[i]
        color = color_lst[i]
        method_path = osp.join(basedir, f"{method}.npy")
        data = np.load(method_path, allow_pickle=True).item()

        subset_score = data["subset_score"]
        subset1_score = data["subset1_score"]
        subset2_score = data["subset2_score"]
        subset3_score = data["subset3_score"]

        draw_bar(
            ax2,
            0 * (num + 1.5) * delta + xdis,
            subset1_score,
            bar_width,
            seg_score=seg_score1,
            color=color,
            label=None,
        )
        draw_bar(
            ax2,
            1 * (num + 1.5) * delta + xdis,
            subset2_score,
            bar_width,
            seg_score=seg_score1,
            color=color,
            label=None,
        )
        draw_bar(
            ax2,
            2 * (num + 1.5) * delta + xdis,
            subset3_score,
            bar_width,
            seg_score=seg_score1,
            color=color,
            label=None,
        )
        draw_bar(
            ax2,
            3 * (num + 1.5) * delta + xdis,
            subset_score,
            bar_width,
            seg_score=seg_score1,
            color=color,
            label=label,
        )

    ax2.set_xticks(
        [
            0 * (num + 1.5) * delta,
            1 * (num + 1.5) * delta,
            2 * (num + 1.5) * delta,
            3 * (num + 1.5) * delta,
        ]
    )
    ax2.set_xticklabels(["subset1", "subset2", "subset3", "whole subset"], fontsize=8)
    yarr = np.arange(0, 1.0, 0.1)
    ax2.set_yticks(yarr)
    ys = []
    for y in yarr:
        ys.append("{:.1f}".format(y))
    ax2.set_yticklabels(ys, fontsize=8)

    ax2.set_ylim((0, yarr[-1]))
    ax2.set_ylabel("Rate", size=10)
    ax2.legend(loc="upper right", fontsize=8, ncol=2)

    ax2.set_title(
        f"Cumulative distribution rate of four subsets at fixed correlation score",
        fontsize=8,
    )

    plt.savefig(f"./results/cumsum.png", dpi=500, bbox_inches="tight")
