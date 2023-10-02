from pathlib import Path
import logging
from torch.utils.data import DataLoader
from data.loader import load_dataset_bb
from utils import init_model, config_logging, get_parser, seed_everything, linear_probe, save_results

config_logging()

parser = get_parser()
args = parser.parse_args()
print(args)

seed_everything(args.seed)

load_roi = 'visualbert' in args.encoder_m or 'lxmert' in args.encoder_m
dataset_path = Path('/data1/datasets/relationship')
train_set, test_set = load_dataset_bb(dataset_path, load_roi=load_roi)
train_loader = DataLoader(train_set, batch_size=args.bs, collate_fn=list)
test_loader = DataLoader(test_set, batch_size=args.bs, collate_fn=list)

model = init_model(args)

f1s_dict, flags_dict = linear_probe(train_loader, test_loader, model, args.k, args.C, args.seed)
logging.info(f"f1_t: {f1s_dict['t']:2.1%}, f1_v: {f1s_dict['v']:2.1%}, f1_tv: {f1s_dict['tv']:2.1%}")

encoder_id = args.encoder_m if args.encoder_m else f'{args.encoder_t}+{args.encoder_v}'
model_id = f'{encoder_id}(0)_k{args.k}'
results = {
    'config': vars(args),
    'f1s_t': [f1s_dict['t']],
    'f1s_v': [f1s_dict['v']],
    'f1s_tv': [f1s_dict['tv']],
    'flags_t': [flags_dict['t']],
    'flags_v': [flags_dict['v']],
    'flags_tv': [flags_dict['tv']],
}
save_results(f'log/linear-probe_{model_id}.json', results)
