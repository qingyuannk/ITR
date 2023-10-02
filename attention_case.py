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
args = parser.parse_args()
args.raw = True
args.bs = 1
cbar_kw = {}

id2sentence = {
    1949: "#22weeks today! Little sister is growing like a weed. #may2016 #babybump",
    3627: '90% dosage reduction of opiates – NO BIG DEAL?',
    4246: "Let's go Wild! @mnwild #mnwild #itsplayoffseason",
    4424: 'RT @AgentCarterTV: #AgentCarter fans are on high alert!',
}

id2threshold = {
    1949: {'vmin': None, 'vmax': None},
    3627: {'vmin': None, 'vmax': None},
    4246: {'vmin': None, 'vmax': None},
    4424: {'vmin': None, 'vmax': None},
}

def normalize_text(text: str):
    # remove the ending URL which is not part of the text
    url_re = r' http[s]?://t.co/\w+'
    text = re.sub(url_re, '', text)
    rt_re = r'^RT @.*: '
    text = re.sub(rt_re, '', text)
    return text


def heatmap(data, row_labels, ax, vmin, vmax):
    if not ax:
        ax = plt.gca()

    ax.imshow(data, cmap='jet', vmin=vmin, vmax=vmax)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)
    ax.tick_params(top=False, bottom=False, labeltop=False, labelbottom=False)
    ax.tick_params(left=False, right=False, labelleft=False, labelright=True)
    plt.setp(ax.get_yticklabels(), ha='left', va='center', size=320//data.shape[0])
    if not args.state_dict:
        ax.spines[:].set_visible(False)
        # ax.grid(which="minor", color="w", linestyle='-', linewidth=3)


if __name__ == '__main__':
    result_path = f'att/{args.state_dict}'
    os.makedirs(result_path, exist_ok=True)

    dataset_path = Path('/home/resource/datasets/relationship')
    train_set, test_set = load_dataset_bb(dataset_path)
    dataset = train_set + test_set
    loader = DataLoader(dataset, batch_size=args.bs, collate_fn=list)

    print(f'loading pre-trained model {args.state_dict}.pt')
    model = MyTvModel.from_pretrained(args)
    if args.state_dict:
        state_dict_path = Path('state_dict')
        state_dict = torch.load(state_dict_path / f'{args.state_dict}.pt')
        model.load_state_dict(state_dict)
    model.eval()

    fig, ax = plt.subplots()
    with torch.no_grad():
        for i, sentences in id2sentence.items():
            encoded_input = model.tokenizer(sentences, padding=True, return_tensors='pt').to(model.device)
            outputs = model.encoder_t(**encoded_input, output_attentions=True, return_dict=True)
            weights = outputs.attentions[-1]

            for input_ids, mask, weight in zip(encoded_input['input_ids'], encoded_input['attention_mask'], weights):
                length = torch.sum(mask)
                tokens = model.tokenizer.convert_ids_to_tokens(input_ids[:length])
                weight = torch.mean(weight, dim=[0, 1]).unsqueeze(1)
                weight = weight.detach().cpu().numpy()

                tokens = tokens[1:-1]
                weight = weight[1:-1]
                weight[:] /= sum(weight)
                if not args.state_dict:
                    weight[:] = 0

                row_labels = np.array(tokens)
                col_labels = np.array(tokens)
                heatmap(weight, row_labels, ax=ax, **id2threshold[i])
                plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
                plt.savefig(f'{result_path}/{i}.svg', bbox_inches='tight', pad_inches=0)
                plt.savefig(f'{result_path}/{i}.pdf', bbox_inches='tight', pad_inches=0)
                plt.cla()
