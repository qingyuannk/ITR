import logging
import os
from argparse import ArgumentParser
from utils import config_logging

config_logging()

settings = [
    # '--encoder_t bert-base-uncased --encoder_v resnet-101',
    # '--encoder_t bert-base-uncased --encoder_v vit-base-patch16-224-in21k',
    '--encoder_t bertweet-base --encoder_v vit-base-patch16-224-in21k',
    '--encoder_m clip-vit-base-patch32',
    '--encoder_m visualbert-vqa-coco-pre',
    '--encoder_m lxmert-base-uncased',
]
tasks = ['0', '1', '2', '3', '4', '1+4', '2+4', '3+4']

parser = ArgumentParser()
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--k', type=int, default=4, choices=(2, 4))
parser.add_argument('--task_ids', type=str, default=None,
                    choices=('1', '2', '3', '4', '1+4', '2+4', '3+4'))
parser.add_argument('--setting', type=str, default='--encoder_m visualbert-vqa-coco-pre')
args = parser.parse_args()


def loop_encoders():
    for setting in settings:
        command = f'python pretrain.py --cuda {args.cuda} --task_ids {args.task_ids} --k {args.k} {setting}'
        logging.info(command)
        os.system(command)


def loop_tasks():
    for task in tasks:
        # command = f'python pretrain.py --cuda {args.cuda} --task_ids {task} --k {args.k} {args.setting}'
        command = f'python finetune.py --cuda {args.cuda} --task_ids {task} --k {args.k} {args.setting}'
        logging.info(command)
        os.system(command)


if __name__ == '__main__':
    # loop_encoders()
    loop_tasks()
