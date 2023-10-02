import logging
from random import shuffle
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from data.loader import load_dataset_100k, load_dataset_bb, load_dataset_fakenews
from model.model_twotower import MyTwoTowerModel
from utils import config_logging, get_parser, seed_everything, save_results, init_model
from utils import encode, cluster, train, linear_probe, evaluate, evaluate_fakenews


config_logging()

parser = get_parser()
parser.add_argument('--size', type=int, default=100000)
parser.add_argument('--subsize', type=int, default=5000)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--preprocess', type=str, default=None, choices=(None, 'normalize', 'scale'))
parser.add_argument('--itr', action='store_true', default=True)
parser.add_argument('--news', action='store_true', default=False)
args = parser.parse_args()
logging.info(args)

encoder_id = args.encoder_m if args.encoder_m else f'{args.encoder_t}+{args.encoder_v}'
model_id = f'{encoder_id}({args.task_ids})_k{args.k}'

seed_everything(args.seed)

load_roi = 'visualbert' in args.encoder_m or 'lxmert' in args.encoder_m
twitter100k_path = Path('/data1/datasets/twitter100k')
relationship_path = Path('/data1/datasets/relationship')
fakenews_path = Path('/data1/datasets/fakenews')
train_set = load_dataset_100k(twitter100k_path, args.size, load_roi=load_roi)
itr_train_set, itr_test_set = load_dataset_bb(relationship_path, load_roi=load_roi)
fakenews_set = load_dataset_fakenews(fakenews_path)
itr_train_loader = DataLoader(itr_train_set, batch_size=args.bs, collate_fn=list)
itr_test_loader = DataLoader(itr_test_set, batch_size=args.bs, collate_fn=list)
fakenews_loader = DataLoader(fakenews_set, batch_size=args.bs, collate_fn=list)

num_iter = args.size // args.subsize
indices_list = [list(range(i * args.subsize, (i + 1) * args.subsize)) for i in range(num_iter)]

logging.info('start loading model')
model = init_model(args)
if isinstance(model, MyTwoTowerModel):
    encoder_t_parameters_id = list(map(id, model.encoder_t.parameters()))
    encoder_v_parameters_id = list(map(id, model.encoder_v.parameters()))
    pretrained_parameters_id = encoder_t_parameters_id + encoder_v_parameters_id
    params = [
        {'params': model.encoder_t.parameters(), 'lr': args.lr},
        {'params': model.encoder_v.parameters(), 'lr': args.lr},
    ]
else:
    pretrained_parameters_id = list(map(id, model.encoder.parameters()))
    params = [
        {'params': model.encoder.parameters(), 'lr': args.lr},
    ]
other_parameters = filter(lambda p: id(p) not in pretrained_parameters_id, model.parameters())
params.append({'params': other_parameters, 'lr': args.lr * 100})
optimizer = getattr(torch.optim, args.optim)(params)

fuse_map = {'1': 't+v', '2': 't+o', '3': 't+c'}
zero_shot = False
cluster_id = None
for task_id in args.task_ids:
    if task_id in fuse_map:
        cluster_id = task_id
        zero_shot = True
        model.classifier = getattr(model, f'head{cluster_id}')

f1s_dict_linear = {'t': [], 'v': [], 'tv': []}
f1s_dict_zero = {'t': [], 'v': [], 'tv': []}
flags_dict_linear = {'t': [], 'v': [], 'tv': []}
flags_dict_zero = {'t': [], 'v': [], 'tv': []}
aucs = []

for epoch in range(args.epochs):
    shuffle(train_set.pairs)

    for i, indices in enumerate(indices_list):
        iteration = epoch * num_iter + i + 1
        logging.info(f'iteration #{iteration:02d}')

        subset = Subset(train_set, indices)
        train_loader = DataLoader(subset, batch_size=args.bs, collate_fn=list, shuffle=True)

        if cluster_id:
            logging.info('start encoding')
            encode(train_loader, model, fuse_map[cluster_id])
            logging.info('start clustering')
            cluster(subset, args.k, cluster_id, args.seed, args.preprocess)
            getattr(model, f'head{cluster_id}').reset_parameters()

        logging.info('start training')
        loss = train(train_loader, model, F.cross_entropy, optimizer, args.task_ids)

        if args.itr:
            logging.info('start evaluating on ITR dataset (linear-probe)')
            f1s_linear, flags_linear = linear_probe(itr_train_loader, itr_test_loader, model, args.k, args.C, args.seed)
            for task in ('t', 'v', 'tv'):
                f1s_dict_linear[task].append(f1s_linear[task])
                flags_dict_linear[task].append(flags_linear[task])
            logging.info(f"LINEAR PROBE: {f1s_linear['t']:2.1%} | {f1s_linear['v']:2.1%} | {f1s_linear['tv']:2.1%}")

            if zero_shot:
                logging.info('start evaluating on ITR dataset (zero-shot)')
                f1s_zero, flags_zero = evaluate(model, itr_test_loader, args.k, category_mapping=True)
                for task in ('t', 'v', 'tv'):
                    f1s_dict_zero[task].append(f1s_zero[task])
                    flags_dict_zero[task].append(flags_zero[task])
                logging.info(f"   ZERO SHOT: {f1s_zero['t']:2.1%} | {f1s_zero['v']:2.1%} | {f1s_zero['tv']:2.1%}")

        if args.news:
            logging.info('start evaluating on fake news dataset (zero-shot)')
            auc = evaluate_fakenews(model, fakenews_loader)
            aucs.append(auc)
            logging.info(f'AUROC on fakenews dataset: {auc:.2f}')

        for pair in subset:
            pair.clean()

if args.itr:
    logging.info(f"best F1 on ITR dataset (linear-probe): {max(f1s_dict_linear['t']):2.1%} | "
                 f"{max(f1s_dict_linear['v']):2.1%} | {max(f1s_dict_linear['tv']):2.1%}")
    if zero_shot:
        logging.info(f"best F1 on ITR dataset    (zero-shot): {max(f1s_dict_zero['t']):2.1%} | "
                     f"{max(f1s_dict_zero['v']):2.1%} | {max(f1s_dict_zero['tv']):2.1%}")

if args.news:
    logging.info(f"best AUROC on fake news dataset (zero-shot): {max(aucs):.2f}")

if args.save:
    results_linear = {
        'config': vars(args),
        'f1s_t': f1s_dict_linear['t'],
        'f1s_v': f1s_dict_linear['v'],
        'f1s_tv': f1s_dict_linear['tv'],
        'flags_t': flags_dict_linear['t'],
        'flags_v': flags_dict_linear['v'],
        'flags_tv': flags_dict_linear['tv'],
    }
    save_results(f'log/linear-probe_{model_id}_{args.seed}.json', results_linear)

    if zero_shot:
        results_zero = {
            'config': vars(args),
            'f1s_t': f1s_dict_zero['t'],
            'f1s_v': f1s_dict_zero['v'],
            'f1s_tv': f1s_dict_zero['tv'],
            'aucs': aucs,
            'flags_t': flags_dict_zero['t'],
            'flags_v': flags_dict_zero['v'],
            'flags_tv': flags_dict_zero['tv'],
        }
        save_results(f'log/zero-shot_{model_id}.json', results_zero)

    torch.save(model.state_dict(), f'state_dict/{model_id}_{args.seed}.pt')
