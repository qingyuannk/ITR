import os
import re
from tqdm import tqdm
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt

from data.loader import load_dataset_bb
from model.model_tv import MyTvModel
from utils import get_parser


parser = get_parser()
parser.add_argument('--state_dict', type=str, default=None)
parser.add_argument('--aggregate', action='store_true', default=True)
parser.add_argument('--remove', action='store_true', default=True)
args = parser.parse_args()
args.raw = True
args.bs = 1


def normalize_text(text: str):
    # remove the ending URL which is not part of the text
    url_re = r' http[s]?://t.co/\w+'
    text = re.sub(url_re, '', text)
    # rt_re = r'^RT @.*: '
    # text = re.sub(rt_re, '', text)
    return text


def heatmap(data, row_labels, ax=None):
    if not ax:
        ax = plt.gca()

    imshow_args = {'cmap': 'jet',   'vmin': 0, 'vmax': 1.1 * np.max(data)} if args.state_dict else \
                  {'cmap': 'Greys', 'vmin': 0, 'vmax': 1}
    ax.imshow(data, **imshow_args)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)
    ax.tick_params(top=False, bottom=False, labeltop=False, labelbottom=False)
    ax.tick_params(left=False, right=False, labelleft=False, labelright=True)
    plt.setp(ax.get_yticklabels(), ha='left', va='center', size=320//data.shape[0])
    ax.spines[:].set_visible(False)


if __name__ == '__main__':
    result_path = f'att/{args.state_dict}'
    os.makedirs(result_path, exist_ok=True)

    dataset_path = Path('/home/resource/datasets/relationship')
    train_set, test_set = load_dataset_bb(dataset_path)
    dataset = train_set + test_set
    loader = DataLoader(dataset, batch_size=args.bs, collate_fn=list)

    model = MyTvModel.from_pretrained(args)
    if args.state_dict:
        print(f'loading pre-trained model {args.state_dict}.pt')
        state_dict_path = Path('state_dict')
        state_dict = torch.load(state_dict_path / f'{args.state_dict}.pt')
        model.load_state_dict(state_dict)
    model.eval()

    fig, ax = plt.subplots()
    with torch.no_grad():
        # for i, batch in enumerate(tqdm(loader)):
        tweet_ids = [4246, 1949, 3627, 4424]
        for tweet_id in tqdm(tweet_ids):
            batch = [dataset[tweet_id]]
            sentences = [normalize_text(pair.text.data) for pair in batch]
            encoded_input = model.tokenizer(sentences, padding=True, return_tensors='pt').to(model.device)
            outputs = model.encoder_t(**encoded_input, output_attentions=True, return_dict=True)
            weights = outputs.attentions[-1]

            for input_ids, mask, weight in zip(encoded_input['input_ids'], encoded_input['attention_mask'], weights):
                length = torch.sum(mask)
                tokens = model.tokenizer.convert_ids_to_tokens(input_ids[:length])
                weight = torch.mean(weight, dim=[0, 1]).unsqueeze(1)
                weight = weight.detach().cpu().numpy()

                # print(model.tokenizer.decode(input_ids))
                # print(tokens)

                if args.aggregate:
                    def end_with(s, suffix):
                        return s[-len(suffix):] == suffix

                    token_list, weight_list = [], []
                    i, n = 0, len(tokens)
                    while i < n:
                        if end_with(tokens[i], '@@'):
                            j = i
                            while end_with(tokens[j], '@@'):
                                j += 1
                            pieces = [re.sub(r'@@$', '', token) for token in tokens[i:j+1]]
                            token_list.append(''.join(pieces))
                            weight_list.append(sum(weight[i:j+1]))
                            i = j
                        elif tokens[i] == '<unk>':
                            pass
                        else:
                            token_list.append(tokens[i])
                            weight_list.append(weight[i])

                        i += 1

                    assert len(token_list) == len(weight_list)
                    tokens = token_list
                    weight = np.array(weight_list)

                if args.remove:
                    tokens = tokens[1:-1]
                    weight = weight[1:-1]

                if args.aggregate or args.remove:
                    weight[:] /= sum(weight)

                # print(len(tokens))
                EPS = 1e-6
                assert 1 - EPS < sum(weight) < 1 + EPS

                if args.state_dict is None:
                    weight[:] = 0

                row_labels = np.array(tokens)
                col_labels = np.array(tokens)
                heatmap(weight, row_labels, ax=ax)
                plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
                plt.savefig(f'{result_path}/{tweet_id}.pdf', bbox_inches='tight', pad_inches=0)
                plt.cla()

            for pair in batch:
                pair.clean()
            torch.cuda.empty_cache()
