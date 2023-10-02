import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data.loader import load_dataset_bb
from model.model_twotower import MyTwoTowerModel
from utils import init_model, config_logging, get_parser, seed_everything, train, evaluate_quad, save_results

config_logging()

parser = get_parser()
args = parser.parse_args()
args.epochs = 10
args.dropout = 0.5
logging.info(args)

seed_everything(args.seed)

load_roi = 'visualbert' in args.encoder_m or 'lxmert' in args.encoder_m
dataset_path = Path('/data1/datasets/relationship')
train_set, test_set = load_dataset_bb(dataset_path, load_roi=load_roi)
train_loader = DataLoader(train_set, batch_size=args.bs, collate_fn=list)
test_loader = DataLoader(test_set, batch_size=args.bs, collate_fn=list)

logging.info('start loading model')
model = init_model(args)
encoder_id = args.encoder_m if args.encoder_m else f'{args.encoder_t}+{args.encoder_v}'
model_id = f'{encoder_id}({args.task_ids})_k{args.k}'
if args.task_ids != '0':
    state_dict_path = Path('state_dict')
    state_dict = torch.load(state_dict_path / f'{model_id}_{args.seed}.pt')
    model.load_state_dict(state_dict)
    model.classifier.reset_parameters()

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

f1s_t, f1s_v, f1s_tv = [], [], []
flags_t, flags_v, flags_tv = [], [], []
for epoch in range(1, args.epochs + 1):
    loss = train(train_loader, model, F.cross_entropy, optimizer, '0')

    f1_dict, flags_dict = evaluate_quad(model, test_loader, category_mapping=False)
    logging.info(f"epoch #{epoch:02d}, loss: {loss:.2f}, "
                 f"f1_t: {f1_dict['t']:2.1%}, f1_v: {f1_dict['v']:2.1%}, f1_tv: {f1_dict['tv']:2.1%}")
    f1s_t.append(f1_dict['t']), f1s_v.append(f1_dict['v']), f1s_tv.append(f1_dict['tv'])
    flags_t.append(flags_dict['t']), flags_v.append(flags_dict['v']), flags_tv.append(flags_dict['tv'])

results = {
    'config': vars(args),
    'f1s_t': f1s_t,
    'f1s_v': f1s_v,
    'f1s_tv': f1s_tv,
    'flags_t': flags_t,
    'flags_v': flags_v,
    'flags_tv': flags_tv,
}
save_results(f'log/fine-tune_{model_id}_{args.seed}.json', results)
