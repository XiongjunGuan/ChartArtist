"""
Description: 
Author: Xiongjun Guan
Date: 2023-09-20 20:12:56
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2023-09-20 20:42:46

Copyright (C) 2023 by Xiongjun Guan, Tsinghua University. All rights reserved.
"""

import numpy as np
import os.path as osp
import scipy.io as scio
import matplotlib.pyplot as plt
import pandas as pd
import os
import os.path as osp
import matplotlib.patches as mpatches


def DrawImporveBoxplot(
    ax, improve, ns, x, width, add_w=0, color="r", label=None, patch_artist=False
):
    """Draw boxplot of improvement scores.

    Args:
        ax (_type_): _description_.
        improve (_type_): improved score.
        ns (_type_): score segmentation.
        x (_type_): original score.
        width (_type_): bar width.
        add_w (int, optional):  Used to adjust relative position. Defaults to 0.
        color (str, optional): bar color. Defaults to "r".
        label (_type_, optional): bar label. Defaults to None.
        patch_artist (bool, optional): fill mode. Defaults to False.

    Returns:
        _type_: _description_
    """

    percent_lst = []
    for i in range(len(ns) - 1):
        ch = (x >= ns[i]) & (x < ns[i + 1])
        im_ch = improve[ch]
        name = str(ns[i]) + "-" + str(ns[i + 1])
        df = pd.DataFrame({name: list(im_ch)})

        percent_lst.append(100 * np.sum(ch) / len(x))

        if patch_artist is False:
            ax.boxplot(
                df,
                showfliers=False,
                patch_artist=True,
                positions=[i + add_w - width * 3],
                widths=width - 0.04,
                medianprops={"color": color},
                boxprops={"color": color, "facecolor": "white"},
                whiskerprops={"color": color},
                capprops={"color": color},
                flierprops={"color": color, "markeredgecolor": color},
            )
        else:
            ax.boxplot(
                df,
                showfliers=False,
                patch_artist=patch_artist,
                positions=[i + add_w - width * 3],
                widths=width - 0.04,
                medianprops={"color": "white"},
                boxprops={"color": color, "facecolor": color},
                whiskerprops={"color": color},
                capprops={"color": color},
                flierprops={"color": color, "markeredgecolor": color},
            )

    return percent_lst


def make_match_name(search_lst, query_lst):
    """Translate sample name to {search name}-{query name}.

    Args:
        search_lst (_type_): _description_
        query_lst (_type_): _description_

    Returns:
        _type_: _description_
    """
    lst = []
    for i in range(len(search_lst)):
        search_name = search_lst[i].replace(" ", "")
        query_name = query_lst[i].replace(" ", "")
        lst.append("{}-{}".format(search_name, query_name))
    return lst


def get_improvement(dst_path, ori_gms, ori_lst):
    """Calculate improvement from the original method to current method.

    Args:
        dst_path (_type_): current method path.
                        A dict contains {"gms_search_name": ..., "gms_query_name": ..., "gms": ...}.
        ori_gms (_type_): score of original method.
        ori_lst (_type_): name of original method.

    Returns:
        _type_: _description_
    """
    data = scio.loadmat(dst_path)
    lst = make_match_name(data["gms_search_name"], data["gms_query_name"])
    gms = data["gms"].reshape((-1,))
    dst_improve = []
    dst_x = []
    for i in range(len(ori_lst)):
        ori_name = ori_lst[i]
        try:
            idx = lst.index(ori_name)
            dst_improve.append(gms[idx] - ori_gms[i])
            dst_x.append(ori_gms[i])
        except:
            continue
    dst_improve = np.array(dst_improve).reshape((-1,))
    return dst_improve, dst_x


if __name__ == "__main__":

    save_dir = "./results/"
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    fig = plt.figure()
    width = 0.1

    ### method 1
    ax1 = fig.add_axes([0, 0, 1, 1])
    data_dir = "./test_data/boxplot/verifinger/"

    data = scio.loadmat(osp.join(data_dir, "original.mat"))
    ori_gms = data["gms"].reshape((-1,))
    ori_lst = make_match_name(data["gms_search_name"], data["gms_query_name"])
    ns1 = list(np.arange(0, 210, 30))

    dst_y, dst_x = get_improvement(osp.join(data_dir, "Gu.mat"), ori_gms, ori_lst)
    percent1_lst = DrawImporveBoxplot(
        ax1, dst_y, ns1, dst_x, width, add_w=0, color="orange", patch_artist=True
    )

    dst_y, dst_x = get_improvement(
        osp.join(data_dir, "Dabouei_timg.mat"), ori_gms, ori_lst
    )
    DrawImporveBoxplot(
        ax1, dst_y, ns1, dst_x, width, add_w=width, color="g", patch_artist=True
    )

    dst_y, dst_x = get_improvement(
        osp.join(data_dir, "DDRNet_timg.mat"), ori_gms, ori_lst
    )
    DrawImporveBoxplot(
        ax1, dst_y, ns1, dst_x, width, add_w=width * 2, color="b", patch_artist=True
    )

    dst_y, dst_x = get_improvement(
        osp.join(data_dir, "DDRNet_withOri_timg.mat"), ori_gms, ori_lst
    )
    DrawImporveBoxplot(
        ax1, dst_y, ns1, dst_x, width, add_w=width * 3, color="m", patch_artist=True
    )

    ### method 2
    ax2 = ax1.twinx()
    data_dir = "./test_data/boxplot/mcc/"
    data = scio.loadmat(osp.join(data_dir, "original.mat"))
    ori_gms = data["gms"].reshape((-1,))
    ori_lst = make_match_name(data["gms_search_name"], data["gms_query_name"])
    ns2 = list(np.arange(0, 560, 80))

    dst_y, dst_x = get_improvement(osp.join(data_dir, "Gu.mat"), ori_gms, ori_lst)
    percent2_lst = DrawImporveBoxplot(
        ax2,
        dst_y,
        ns2,
        dst_x,
        width,
        add_w=width * 4,
        color="orange",
        patch_artist=False,
    )

    dst_y, dst_x = get_improvement(
        osp.join(data_dir, "Dabouei_timg.mat"), ori_gms, ori_lst
    )
    DrawImporveBoxplot(
        ax2, dst_y, ns2, dst_x, width, add_w=width * 5, color="g", patch_artist=False
    )

    dst_y, dst_x = get_improvement(
        osp.join(data_dir, "DDRNet_timg.mat"), ori_gms, ori_lst
    )
    DrawImporveBoxplot(
        ax2, dst_y, ns2, dst_x, width, add_w=width * 6, color="b", patch_artist=False
    )

    dst_y, dst_x = get_improvement(
        osp.join(data_dir, "DDRNet_withOri_timg.mat"), ori_gms, ori_lst
    )
    DrawImporveBoxplot(
        ax2, dst_y, ns2, dst_x, width, add_w=width * 7, color="m", patch_artist=False
    )

    ### labels
    xlabels = []
    for i in range(len(ns1) - 1):
        xlabels.append("{}-{}\n({:.1f}%)".format(ns1[i], ns1[i + 1], percent1_lst[i]))
    xlabels = tuple(xlabels)
    ax1.set_xticks(
        np.arange(len(ns1) - 1) + 0.5 * width,
        xlabels,
        rotation=0,
        fontsize=13,
    )

    xlabels = []
    for i in range(len(ns2) - 1):
        xlabels.append("{}-{}\n({:.1f}%)".format(ns2[i], ns2[i + 1], percent2_lst[i]))
    xlabels = tuple(xlabels)
    secax1 = ax1.secondary_xaxis("top", functions=(lambda x: x, lambda y: y))
    secax1.set_xticks(
        np.arange(len(ns2) - 1) + 0.5 * width,
        xlabels,
        rotation=0,
        fontsize=13,
    )

    ax1.tick_params(axis="y", labelsize=13)
    ax2.tick_params(axis="y", labelsize=13)
    ax2.set_ylim([-360, 840])
    ax1.set_ylim([-180, 420])
    ax1.grid(axis="y")

    # ax2.tick_params(axis="y", colors="gray")
    # secax1.tick_params(axis="x", colors="gray")

    ax1.set_ylabel("Imporved Matching Score by VeriFinger", fontsize=13)
    ax1.set_xlabel("Matching Score without Rectification by VeriFinger", fontsize=13)
    ax2.set_ylabel("Imporved Matching Score by MCC", fontsize=13)
    secax1.set_xlabel("Matching Score without Rectification by MCC", fontsize=13)

    patches = [
        mpatches.Patch(color="orange", label="Gu et al. :   VeriFinger"),
        mpatches.Patch(color="g", label="Dabouei et al. :   VeriFinger"),
        mpatches.Patch(color="b", label="Proposed method :   VeriFinger"),
        mpatches.Patch(color="m", label="Proposed method +O :   VeriFinger"),
        mpatches.Patch(edgecolor="orange", facecolor="white", label="MCC"),
        mpatches.Patch(edgecolor="g", facecolor="white", label="MCC"),
        mpatches.Patch(edgecolor="b", facecolor="white", label="MCC"),
        mpatches.Patch(edgecolor="m", facecolor="white", label="MCC"),
    ]
    ax1.legend(
        handles=patches, loc="upper right", ncol=2, fontsize=11, markerfirst=False
    )

    ax1.set_title(
        "Minutiae Based\nScore Curves on\nTDF_V2_T",
        fontsize=13,
        bbox=dict(ec="black", fc="white"),
        loc="left",
        y=0.84,
        x=0.05,
    )

    plt.savefig(
        osp.join(save_dir, "boxplot.png"),
        dpi=500,
        bbox_inches="tight",
    )
