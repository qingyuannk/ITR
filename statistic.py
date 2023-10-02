import json
import argparse
from pathlib import Path
import numpy as np
from collections import defaultdict
from numpy import argmax


def main(args, variables, setting='_'):
    log_path = Path('log')
    filenames = [str(filename) for filename in log_path.iterdir()]
    filenames = filter(lambda filename: setting in filename, filenames)
    filenames = filter(lambda filename: f'k{args.k}' in filename, filenames)
    filenames = filter(lambda filename: args.eval in filename, filenames)
    filenames = sorted(filenames)
    results = defaultdict(dict)

    for variable in variables:
        for task in ('t', 'v', 'tv'):
            results[task][variable] = 0.0
        for filename in filenames:
            if variable not in str(filename):
                continue
            print(f'reading {filename}')

            with open(filename, 'r') as json_file:
                result = json.load(json_file)
                for task in ('t', 'v', 'tv'):
                    if args.iteration != 0:
                        iteration = args.iteration - 1
                    elif args.k == 2 and args.eval == 'zero-shot':
                        iteration = argmax(result[f'f1s_{task}'])
                    else:
                        iteration = argmax(result['f1s_tv'])
                    results[task][variable] = result[f'f1s_{task}'][iteration]

    print()
    print(setting)
    avg_f1_t, avg_f1_v, avg_f1_tv = [], [], []
    cnt = 0
    for variable in variables:
        f1_t, f1_v, f1_tv = results['t'][variable], results['v'][variable], results['tv'][variable]
        if f1_t != 0.0:
            cnt += 1
            avg_f1_t.append(f1_t)
            avg_f1_v.append(f1_v)
            avg_f1_tv.append(f1_tv)
            print(variable, end='\t')
            print(f'{f1_t * 100:.1f} {f1_v * 100:.1f} {f1_tv * 100:.1f}')
    
    print(f'avg_f1_t {np.mean(avg_f1_t) * 100:.1f} avg_f1_v {np.mean(avg_f1_v) * 100:.1f} avg_f1_tv {np.mean(avg_f1_tv) * 100:.1f}')
    print(f'std_f1_t {np.std(avg_f1_t) * 100:.1f} std_f1_v {np.std(avg_f1_v) * 100:.1f} std_f1_tv {np.std(avg_f1_tv) * 100:.1f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', default=0, type=int)
    parser.add_argument('--k', type=int, default=4, choices=(2, 4))
    parser.add_argument('--eval', type=str, default='linear-probe', choices=('linear-probe', 'zero-shot', 'fine-tune'))
    parser.add_argument('--setting', type=str, default='bertweet-base+vit-base-patch16-224-in21k')
    args = parser.parse_args()

    tasks = [f"({task})_k4_{seed}" for task in ['1+4'] for seed in [0, 32, 128, 256, 1024, 2048]]
    main(args, tasks, args.setting)
