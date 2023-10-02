import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from numpy import argmax

parser = argparse.ArgumentParser()
parser.add_argument('--iteration', default=-1, type=int)
parser.add_argument('--setting', default='_', type=str)
args = parser.parse_args()

log_path = Path('log')
split = 3576
tweet_ids = [3928, 4195, 4430, 4454]
labels = ['text is not represented & image does not add', 'text is not represented & image adds',
          'text is represented & image does not add', 'text is represented & image adds']

def main(variables, setting='_'):
    filenames = sorted([str(filename) for filename in log_path.iterdir() if setting in str(filename)], reverse=True)

    for variable in variables:
        for filename in filenames:
            if variable not in str(filename):
                continue

            print(f'reading {filename}')
            with open(filename, 'r') as json_file:
                result = json.load(json_file)
                flags = result['flags_tv'][args.iteration]

            for tweet_id in tweet_ids:
                print(f'{tweet_id}: {labels[flags[tweet_id - 3576]]}')
        print('----------------------------------------------------------------------------------------------------')


if __name__ == '__main__':
    # main([f"({task})" for task in ['vanilla', '1', '2', '3', '4', '5', '2+5', '3+5', '4+5']], 'linear-probe')
    main([f"({task})" for task in ['vanilla', '1', '2', '3', '4', '5', '2+5', '3+5', '4+5']], 'fine-tune')
    # main([f"({task})" for task in ['2', '3', '4', '2+5', '3+5', '4+5']], 'zero-shot')
