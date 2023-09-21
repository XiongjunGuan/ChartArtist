"""
Description: 
Author: Xiongjun Guan
Date: 2023-09-20 20:41:05
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2023-09-20 20:42:21

Copyright (C) 2023 by Xiongjun Guan, Tsinghua University. All rights reserved.
"""

import scipy.io as scio
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = scio.loadmat("./test_data/line_with_bar.mat")
    err_gt = data["err_gt"].reshape((-1,))
    err_Gu = data["err_Gu"].reshape((-1,))
    err_Dabouei = data["err_Dabouei"].reshape((-1,))
    err_Unet = data["err_Unet"].reshape((-1,))
    err_My = data["err_My"].reshape((-1,))
    params = data["params"].reshape((-1,))

    fig = plt.figure()
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height])
    ax.plot(
        err_gt[:-1],
        err_gt[:-1],
        color="b",
        linestyle="-",
        marker="^",
        linewidth=2,
        label="No Rectification",
        markerfacecolor="none",
        markersize=5,
    )
    ax.plot(
        err_gt[:-1],
        err_Gu[:-1],
        color="r",
        linestyle="-",
        marker="+",
        linewidth=2,
        label="Gu et al.",
        markerfacecolor="none",
        markersize=5,
    )
    ax.plot(
        err_gt[:-1],
        err_Dabouei[:-1],
        color="lime",
        linestyle="-",
        marker="o",
        linewidth=2,
        label="Dabouei et al.",
        markerfacecolor="none",
        markersize=5,
    )
    ax.plot(
        err_gt[:-1],
        err_Unet[:-1],
        color="cyan",
        linestyle="-",
        marker="x",
        linewidth=2,
        label="U-Net",
        markerfacecolor="none",
        markersize=5,
    )
    ax.plot(
        err_gt[:-1],
        err_My[:-1],
        color="m",
        linestyle="-",
        marker="s",
        linewidth=2,
        label="Proposed",
        markerfacecolor="none",
        markersize=5,
    )
    print(err_Gu[-1])
    print(err_Dabouei[-1])
    print(err_Unet[-1])
    print(err_My[-1])
    plt.xlim(3, 27)
    plt.grid()

    plt.legend(loc="lower right", fontsize=8.5, ncol=3)

    left, bottom, width, height = 0.17, 0.57, 0.25, 0.25
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.bar(
        x=0, bottom=5, height=0.7, width=err_gt[-1], orientation="horizontal", color="b"
    )
    ax2.bar(
        x=0, bottom=4, height=0.7, width=err_Gu[-1], orientation="horizontal", color="r"
    )
    ax2.bar(
        x=0,
        bottom=3,
        height=0.7,
        width=err_Dabouei[-1],
        orientation="horizontal",
        color="lime",
    )
    ax2.bar(
        x=0,
        bottom=2,
        height=0.7,
        width=err_Unet[-1],
        orientation="horizontal",
        color="cyan",
    )
    ax2.bar(
        x=0, bottom=1, height=0.7, width=err_My[-1], orientation="horizontal", color="m"
    )
    plt.xlim(5, 17)
    plt.xticks(size=8.5)
    plt.yticks([])

    ax2.set_title("Average Regression Error", fontsize=8.5)

    ax.set_xlabel("Degree of Distortion", fontsize=12)
    ax.set_ylabel("Regression Error", fontsize=12)

    plt.savefig("./results/line_width_bar.png", dpi=500, bbox_inches="tight")
