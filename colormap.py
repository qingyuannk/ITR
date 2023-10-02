import argparse
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--cmap', type=str, default='jet')
args = parser.parse_args()


fig, ax = plt.subplots(figsize=(40, 1))
gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))
ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(args.cmap))
ax.set_axis_off()
plt.savefig(f'{args.cmap}.pdf', bbox_inches='tight', pad_inches=0)
