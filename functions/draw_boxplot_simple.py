'''
Description: 
Author: Xiongjun Guan
Date: 2023-09-20 20:44:53
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2023-09-21 15:40:33

Copyright (C) 2023 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''

import numpy as np
import os.path as osp
import scipy.io as scio
import matplotlib.pyplot as plt
import pandas as pd


def DrawImporveBoxplot(ax, improve, x, width, add_w=0, color="r", label=None):
    """ Draw improvements as boxplot.

    Args:
        ax (_type_): axis of figure.
        improve (_type_): improved score.
        x (_type_): _original score.
        width (_type_): bar width.
        add_w (int, optional): Used to adjust relative position. Defaults to 0.
        color (str, optional): bar color. Defaults to 'r'.
        label (_type_, optional): bar label. Defaults to None.
    """
    n = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    for i in range(len(n) - 1):
        ch = (x >= n[i]) & (x < n[i + 1])
        im_ch = improve[ch]
        name = str(n[i]) + "-" + str(n[i + 1])
        df = pd.DataFrame({name: list(im_ch)})
   
        ax.boxplot(
            df,
            showfliers=False,
            positions=[i + add_w],
            widths=width,
            medianprops={"color": color},
            boxprops=dict(color=color),
            whiskerprops={"color": color},
            capprops={"color": color},
            flierprops={"color": color, "markeredgecolor": color},
        )
        
        plt.xticks([])
        plt.xticks(
            np.arange(len(n) - 1) + 0.5 * width,
            (
                "0-50",
                "50-100",
                "100-150",
                "150-200",
                "200-250",
                "250-300",
                "300-350",
                "350-400",
                "400-450",
                "450-500",
            ),
            rotation=45,
        )
    ax.plot([0,2],[-500,-500],color=color,label=label)


if __name__ == "__main__":
    data_dir = "./test_data/boxplot_simple/"
    title = "test"

    data = scio.loadmat(osp.join(data_dir, "vfscore_ori.mat"))
    ori_gms = data["gms"]

    data = scio.loadmat(osp.join(data_dir, "vfscore_Gu.mat"))
    Gu_gms = data["gms"]

    data = scio.loadmat(osp.join(data_dir, "vfscore_Dabouei.mat"))
    Dabouei_gms = data["gms"]

    data = scio.loadmat(osp.join(data_dir, "vfscore_Unet.mat"))
    Unet_gms = data["gms"]

    data = scio.loadmat(osp.join(data_dir, "vfscore_My.mat"))
    My_gms = data["gms"]

    Gu_improve = Gu_gms - ori_gms
    Dabouei_improve = Dabouei_gms - ori_gms
    Unet_improve = Unet_gms - ori_gms
    My_improve = My_gms - ori_gms

    
    fig, ax = plt.subplots()
    DrawImporveBoxplot(ax, Gu_improve, ori_gms, 0.15, add_w=0, color="r", label="Rectified by Gu et al. Method")
    DrawImporveBoxplot(
        ax, Dabouei_improve, ori_gms, 0.15, add_w=0.15, color="lime", label="Rectified by Dabouei et al. Method"
    )
    DrawImporveBoxplot(ax, Unet_improve, ori_gms, 0.15, add_w=0.3, color="cyan", label="Rectified by U-Net")
    DrawImporveBoxplot(ax, My_improve, ori_gms, 0.15, add_w=0.45, color="m", label="Rectified by Proposed method")

    plt.ylim([-200,480])
    plt.grid(axis='y')
    plt.legend(loc="upper left",ncol=2,fontsize=8.5)
    plt.xlabel("Matching Score without Rectification",fontsize=12)
    plt.ylabel("Improved Matching Score",fontsize=12)

    plt.savefig(
        "./results/boxplot_simple.png",
        dpi=500,
        bbox_inches="tight",
    )

 