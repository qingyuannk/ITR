import json
import argparse
from pathlib import Path
from itertools import permutations
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--setting', default='_', type=str)
args = parser.parse_args()

fig_path = Path('fig')
log_path = Path('log/similarity/20k_subset/')


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbar_label="", **kwargs):
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)


def align_intrinsic(scores, centers):
    m, n = scores.shape

    for i in range(1, m):
        last_center = centers[i - 1]
        min_dist, best_map = float('inf'), list(range(n))

        for permutation in permutations(range(n)):
            this_center = centers[i][np.array(permutation)]
            dist = sum([np.linalg.norm(c1 - c2) for c1, c2 in zip(this_center, last_center)])
            if dist < min_dist:
                min_dist = dist
                best_map = np.array(permutation)

        scores[i] = scores[i][best_map]
        centers[i] = centers[i][best_map]

    return scores


def align_extrinsic_f1(scores, flags, anchor=0, iteration=0):
    m, n = len(scores), len(scores[0])

    for i in range(m):
        max_f1, best_map = -1.0, list(range(n))
        for permutation in permutations(range(n)):
            flags_tmp = [permutation[flag] for flag in flags[i][iteration]]
            f1 = f1_score(flags_tmp, flags[anchor][iteration], average='weighted')
            if f1 > max_f1:
                max_f1 = f1
                best_map = np.array(permutation)

        scores[i] = scores[i][best_map]
        print(max_f1, best_map)

    return scores


def align_extrinsic_gold(scores, maps, iteration=-1):
    for i in range(len(scores)):
        class_map = np.array(maps[i][iteration])
        scores[i] = scores[i][class_map]
    return scores


if __name__ == '__main__':
    filenames = [f'similarity_bertweet-base+vit-base-patch16-224-in21k({task_ids})_5k_100k_0.json'
                 for task_ids in ['2+5', '3+5', '4+5']]
    fig, axes = plt.subplots(3, 1)
    cbar_labels = [f'task #{i} & #4' for i in (1, 2, 3)]
    cbar_kw = {'vmin': 0, 'vmax': 0.8, 'cmap': 'YlGn'}

    scores_list, map_list = [], []
    for filename in filenames:
        with open(log_path/filename, 'r') as json_file:
            result = json.load(json_file)
            scores, centers, map = result['scores_list'], result['centers_list'], result['map_list']
            scores, centers, map = np.array(scores), np.array(centers), np.array(map)
            scores = align_intrinsic(scores, centers)
            scores_list.append(scores.transpose())
            map_list.append(map)
    scores_list = align_extrinsic_gold(scores_list, map_list)

    for scores, ax, cbar_label in zip(scores_list, axes, cbar_labels):
        row_labels = np.array([f'C{i}' for i in range(1, scores.shape[0] + 1)])
        col_labels = np.array([f'{i}' for i in range(1, scores.shape[1] + 1)])
        heatmap(scores, row_labels, col_labels, ax=ax, **cbar_kw, cbar_label=cbar_label)
        fig.tight_layout()

    plt.savefig(fig_path/'similarity_20k_aligned.pdf')
