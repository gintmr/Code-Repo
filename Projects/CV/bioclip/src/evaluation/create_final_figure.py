import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from evaluation.hierarchy_tree_image import (
    TAXONOMICAL_RANKS,
    _get_hierarchy_lbl_map,
    get_colors,
    mahalanobis,
)


def create_final_image(
    openai_features,
    openai_labels,
    bioclip_features,
    bioclip_labels,
    output="tmp/final",
    top_k=6,
    dpi=100,
):
    os.makedirs(output, exist_ok=True)

    base_colors = get_colors()

    nrows = 4
    ncols = 3
    height = 7
    width = 5
    final_fig = plt.figure(figsize=(height * nrows, width * ncols), dpi=dpi)
    final_spec = gridspec.GridSpec(2, 1)
    row1_spec = gridspec.GridSpecFromSubplotSpec(
        2, 3, subplot_spec=final_spec[0], wspace=0.05
    )
    row2_spec = gridspec.GridSpecFromSubplotSpec(
        2, 3, subplot_spec=final_spec[1], wspace=0.05
    )
    for lvl in range(1, len(openai_labels)):
        oa_feats = openai_features[lvl]
        oa_lbls = openai_labels[lvl]
        bio_feats = bioclip_features[lvl]

        lbl_lengths = []
        sorted_lbls = sorted(list(set(oa_lbls[:, lvl])))
        for lbl in sorted_lbls:
            idx = oa_lbls[:, lvl] == lbl
            lbl_lengths.append([lbl, len(oa_feats[idx])])
        lbl_lengths = sorted(lbl_lengths, key=lambda x: x[1], reverse=True)

        row = (lvl - 1) // ncols
        col = (lvl - 1) % ncols

        if row == 0:
            bio_ax = final_fig.add_subplot(row1_spec[0, col])
            oa_ax = final_fig.add_subplot(row1_spec[1, col])
        else:
            bio_ax = final_fig.add_subplot(row2_spec[0, col])
            oa_ax = final_fig.add_subplot(row2_spec[1, col])

        plt.setp(bio_ax.spines.values(), lw=3, color="black")  # Set border width
        plt.setp(oa_ax.spines.values(), lw=3, color="black")  # Set border width

        highest_preceding_rank = oa_lbls[0, lvl - 1].split("_")[-1]
        fig_title = TAXONOMICAL_RANKS[lvl]
        fig_title += f" of {highest_preceding_rank}"
        bio_ax.set_title(f"(B) {fig_title}", fontsize=25, y=1.02)
        oa_ax.set_title(f"(O) {fig_title}", fontsize=25, y=1.02)

        c = 0
        for lbl in sorted_lbls:
            if top_k > 0:
                if lbl not in np.array(lbl_lengths)[:top_k, 0]:
                    continue
            idx = oa_lbls[:, lvl] == lbl
            bio_feat = bio_feats[idx]
            oa_feat = oa_feats[idx]
            name = f"{lbl.split('_')[-1]}"

            mah_dist = mahalanobis(bio_feat)
            p = 1 - chi2.cdf(mah_dist, bio_feat.shape[1] - 1)
            filtered_idx = p >= 0.001
            bio_plot_feat = bio_feat[filtered_idx]

            mah_dist = mahalanobis(oa_feat)
            p = 1 - chi2.cdf(mah_dist, oa_feat.shape[1] - 1)
            filtered_idx = p >= 0.001
            oa_plot_feat = oa_feat[filtered_idx]

            markersize = 2
            if lvl == 5:
                markersize *= 2
            elif lvl == 6:
                markersize *= 4
            bio_ax.scatter(
                bio_plot_feat[:, 0],
                bio_plot_feat[:, 1],
                label=name,
                color=base_colors[c],
                alpha=0.50,
                s=markersize,
                rasterized=True,
            )  # have to rasterize for .pdf to load quicker
            oa_ax.scatter(
                oa_plot_feat[:, 0],
                oa_plot_feat[:, 1],
                label=name,
                color=base_colors[c],
                alpha=0.50,
                s=markersize,
                rasterized=True,
            )  # have to rasterize for .pdf to load quicker

            c += 1

        markerscale = 8
        if lvl == 5:
            markerscale = 6
        elif lvl == 6:
            markerscale = 4
        bio_ax.legend(loc="upper right", ncols=2, markerscale=markerscale, fontsize=12)
        oa_ax.legend(loc="upper right", ncols=2, markerscale=markerscale, fontsize=12)

        # Turn off axis ticks
        bio_ax.set_xticks([])
        oa_ax.set_xticks([])
        bio_ax.set_yticks([])
        oa_ax.set_yticks([])

    final_spec.tight_layout(figure=final_fig, h_pad=5, w_pad=0.1)
    final_fig.savefig(os.path.join(output, "full_image.png"))
    final_fig.savefig(os.path.join(output, "full_image.pdf"))


def _get_args():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--val_root", type=str, default="/local/scratch/cv_datasets/inat21/raw/val"
    )
    parser.add_argument(
        "--results_output", type=str, default="/local/scratch/carlyn.1/test_output"
    )
    parser.add_argument(
        "--openai_tsne_data",
        type=str,
        default="openai_tsne_rerun_top_6_remove_outliers",
    )
    parser.add_argument(
        "--bioclip_tsne_data",
        type=str,
        default="bioclip_tsne_rerun_top_6_remove_outliers",
    )
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--dpi", type=float, default=500)

    return parser.parse_args()


if __name__ == "__main__":
    args = _get_args()
    hierarchy_label_map = _get_hierarchy_lbl_map(args.val_root)

    openai_features = []
    openai_labels = []
    bioclip_features = []
    bioclip_labels = []

    openai_tsne_dir = os.path.join(args.results_output, args.openai_tsne_data)
    bioclip_tsne_dir = os.path.join(args.results_output, args.bioclip_tsne_data)

    for i in range(7):
        openai_features.append(
            np.load(os.path.join(openai_tsne_dir, f"precomputed_{i}.npy"))
        )
        openai_labels.append(
            np.load(os.path.join(openai_tsne_dir, f"precomputed_{i}_labels.npy"))
        )
        bioclip_features.append(
            np.load(os.path.join(bioclip_tsne_dir, f"precomputed_{i}.npy"))
        )
        bioclip_labels.append(
            np.load(os.path.join(bioclip_tsne_dir, f"precomputed_{i}_labels.npy"))
        )

    output_name = os.path.join(args.results_output, "final")
    os.makedirs(output_name, exist_ok=True)

    for lvl in range(len(openai_labels)):
        for a_lbl, b_lbl in zip(openai_labels[lvl], bioclip_labels[lvl]):
            for i in range(7):
                assert a_lbl[i] == b_lbl[i], "Labels do not match"

    create_final_image(
        openai_features,
        openai_labels,
        bioclip_features,
        bioclip_labels,
        output=output_name,
        top_k=args.top_k,
        dpi=args.dpi,
    )
