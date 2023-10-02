import os
import argparse
import random
import json
import logging
from tqdm import tqdm
from itertools import permutations
import numpy as np
from sklearn.preprocessing import normalize, scale
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
import torch
import torch.nn.functional as F
from model import model_tv, model_visualbert, model_lxmert, model_clip


def config_logging():
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--k', type=int, default=4, choices=(2, 4))
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--C', type=float, default=0.25)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--optim', type=str, default='AdamW', choices=('SGD', 'Adam', 'AdamW'))
    parser.add_argument('--encoder_t', type=str, default='bert-base-uncased',
                        choices=('bert-base-uncased', 'roberta-base', 'bertweet-base'))
    parser.add_argument('--encoder_v', type=str, default='vit-base-patch16-224-in21k',
                        choices=('resnet-101', 'vit-base-patch16-224-in21k'))
    parser.add_argument('--encoder_m', type=str, default='',
                        choices=('', 'visualbert-vqa-coco-pre', 'lxmert-base-uncased',
                                 'clip-vit-base-patch32', 'vilt-b32-mlm'))
    parser.add_argument('--task_ids', type=str, default='0',
                        choices=('0', '1', '2', '3', '4', '1+4', '2+4', '3+4'))
    parser.add_argument('--freeze_t', action='store_true', default=False)
    parser.add_argument('--freeze_v', action='store_true', default=False)
    parser.add_argument('--raw', action='store_true', default=False)
    parser.add_argument('--save', action='store_true', default=True)
    return parser


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def init_model(args):
    if 'visualbert' in args.encoder_m:
        return model_visualbert.MyVisualBertModel(args)
    if 'lxmert' in args.encoder_m:
        return model_lxmert.MyLxmertModel(args)
    if 'clip' in args.encoder_m:
        return model_clip.MyClipModel(args)
    return model_tv.MyTvModel(args)


def encode(loader, model, fuse='t+v'):
    model.eval()
    with torch.no_grad():
        for batch in loader:
            model.encode(batch, fuse)


def cluster(dataset, k, task_id, seed=0, preprocess=None):
    x = torch.stack([pair.embedding for pair in dataset]).cpu().numpy()

    if preprocess:
        x = {
            'normalize': normalize,
            'scale': scale,
        }[preprocess](x, axis=0)

    kmeans = KMeans(n_clusters=k, random_state=seed)
    kmeans.fit(x)

    pseudo_flags = kmeans.labels_
    for pair, pseudo_flag in zip(dataset, pseudo_flags):
        setattr(pair, f'pseudo_flag_{task_id}', int(pseudo_flag))

    return kmeans


def train(loader, model, criteria, optimizer, task_ids):
    model.train()
    device = model.device

    weight_map = {'0': 1.0, '1': 0.2, '2': 1.0, '3': 1.0, '4': 1.0}
    loss_map = {'0': [], '1': [], '2': [], '3': [], '4': []}
    losses = []
    for batch in loader:
        embeddings = model.encode(batch)
        total_loss = torch.zeros(1, device=device)

        for task_id in task_ids:
            if task_id == '+': continue

            if task_ids == '0':  # supervised with ground-truth
                logits = model.classifier(embeddings)
                target = torch.tensor([sample.flag_tv for sample in batch], device=device)
                loss = criteria(logits, target)

            elif task_id in ['1', '2', '3']:  # deep clustering
                logits = getattr(model, f'head{task_id}')(embeddings)
                target = torch.tensor([getattr(pair, f'pseudo_flag_{task_id}') for pair in batch], device=device)
                loss = criteria(logits, target)

            elif task_id == '4':  # contrastive learning
                T = 1.0  # softmax temperature (default: 1.0)

                embeddings_t = torch.stack([pair.text.embedding for pair in batch])
                embeddings_v = torch.stack([pair.image.embedding for pair in batch])
                embeddings_t = model.proj_t(embeddings_t)
                embeddings_v = model.proj_v(embeddings_v)
                embeddings_t = F.normalize(embeddings_t, p=2, dim=1)
                embeddings_v = F.normalize(embeddings_v, p=2, dim=1)

                logits = (embeddings_t @ embeddings_v.T)
                logits /= T
                target = torch.arange(len(batch), device=model.device)

                loss_t = F.cross_entropy(logits, target)
                loss_v = F.cross_entropy(logits.T, target)
                loss = (loss_t + loss_v) / 2

            else:
                raise ValueError(f'Invalid task id: {task_id}')

            loss *= weight_map[task_id]
            total_loss += loss
            print(loss.item())
            loss_map[task_id].append(loss.item())

        losses.append(total_loss.item())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    for task_id in task_ids:
        if task_id in loss_map and loss_map[task_id]:
            logging.info(f'loss of task#{task_id}: {np.mean(loss_map[task_id]):.2f}')

    return np.mean(losses)


def compute_score(true_flags, pred_flags):
    f1_dict, flags_dict = {}, {}
    f1_dict['tv'] = f1_score(true_flags, pred_flags, average='weighted')
    flags_dict['tv'] = pred_flags

    flag_map_t = [0, 0, 1, 1]
    true_flags_t = [flag_map_t[true_flag] for true_flag in true_flags]
    pred_flags_t = [flag_map_t[pred_flag] for pred_flag in pred_flags]
    f1_dict['t'] = f1_score(true_flags_t, pred_flags_t, average='weighted')
    flags_dict['t'] = pred_flags_t

    flag_map_v = [0, 1, 0, 1]
    true_flags_v = [flag_map_v[true_flag] for true_flag in true_flags]
    pred_flags_v = [flag_map_v[pred_flag] for pred_flag in pred_flags]
    f1_dict['v'] = f1_score(true_flags_v, pred_flags_v, average='weighted')
    flags_dict['v'] = pred_flags_v

    return f1_dict, flags_dict


def evaluate(model, loader, k, category_mapping=False):
    if k == 2:
        return evaluate_bin(model, loader, category_mapping)
    if k == 4:
        return evaluate_quad(model, loader, category_mapping)
    raise ValueError(f'Invalid k ({k}), k must be either 2 or 4')


def evaluate_bin(model, loader, category_mapping=False):
    true_flags_t = [pair.flag_t for batch in loader for pair in batch]
    true_flags_v = [pair.flag_v for batch in loader for pair in batch]
    pred_flags = model.predict(loader)

    f1_dict = {'tv': 0.0}
    true_flags_dict = {'t': true_flags_t, 'v': true_flags_v}
    pred_flags_dict = {'t': pred_flags, 'v': pred_flags, 'tv': []}

    for task in ('t', 'v'):
        if category_mapping:
            best_map, best_f1 = None, 0
            for flag_map in permutations(range(2)):
                mapped_flags = [flag_map[pred_flag] for pred_flag in pred_flags]
                f1 = f1_score(true_flags_dict[task], mapped_flags, average='weighted')
                if f1 > best_f1:
                    best_f1 = f1
                    best_map = flag_map
            pred_flags_dict[task] = [best_map[pred_flag] for pred_flag in pred_flags]

        f1_dict[task] = f1_score(true_flags_dict[task], pred_flags_dict[task], average='weighted')

    return f1_dict, pred_flags_dict


def evaluate_quad(model, loader, category_mapping=False):
    true_flags = [pair.flag_tv for batch in loader for pair in batch]
    pred_flags = model.predict(loader)

    best_map, best_f1 = None, 0
    if category_mapping:
        for flag_map in permutations(range(4)):
            mapped_flags = [flag_map[pred_flag] for pred_flag in pred_flags]
            f1 = f1_score(true_flags, mapped_flags, average='weighted')
            if f1 > best_f1:
                best_f1 = f1
                best_map = flag_map
        pred_flags = [best_map[pred_flag] for pred_flag in pred_flags]

    return compute_score(true_flags, pred_flags)


def evaluate_fakenews_fc(model, loader):
    labels = [pair.flag_t for batch in loader for pair in batch]
    scores_list = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            scores = model(batch)
            scores = torch.softmax(scores, dim=1)
            scores_list.append(scores)
    scores_list = torch.concat(scores_list, dim=0)

    best_auc = 0
    for index in range(2):
        scores = scores_list[:, index]
        auc = roc_auc_score(labels, scores)
        best_auc = max(best_auc, auc)

    return best_auc


def evaluate_fakenews(model, loader):
    labels = [pair.flag_t for batch in loader for pair in batch]
    scores = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            model.encode(batch)
            embeddings_t = torch.stack([pair.text.embedding for pair in batch])
            embeddings_v = torch.stack([pair.image.embedding for pair in batch])
            embeddings_t = model.proj_t(embeddings_t)
            embeddings_v = model.proj_v(embeddings_v)
            scores += (1.0 - F.cosine_similarity(embeddings_t, embeddings_v, dim=1)).tolist()

    return roc_auc_score(labels, scores)


def evaluate_cluster(kmeans, dataset, k=4):
    x = torch.stack([pair.embedding for pair in dataset]).cpu().numpy()
    true_flags = [pair.flag_tv for pair in dataset]
    pred_flags = kmeans.predict(x)

    best_f1, best_map = 0, None
    for flag_map in permutations(range(k)):
        mapped_flags = [flag_map[pred_flag] for pred_flag in pred_flags]
        f1 = f1_score(true_flags, mapped_flags, average='weighted')
        if f1 > best_f1:
            best_f1 = f1
            best_map = flag_map
    pred_flags = [best_map[pred_flag] for pred_flag in pred_flags]
    f1_dict, flags_dict = compute_score(true_flags, pred_flags)

    return f1_dict, flags_dict, best_map


def linear_probe(train_loader, test_loader, model, k, C=1.0, seed=0):
    print(k)
    if k == 2:
        return linear_probe_bin(train_loader, test_loader, model, C, seed)
    if k == 4:
        return linear_probe_quad(train_loader, test_loader, model, C, seed)
    raise ValueError(f'Invalid k ({k}), k must be either 2 or 4')


def linear_probe_bin(train_loader, test_loader, model, C=1.0, seed=0):
    encode(train_loader, model)
    encode(test_loader, model)

    x_train = torch.stack([pair.embedding for batch in train_loader for pair in batch]).cpu().numpy()
    x_test = torch.stack([pair.embedding for batch in test_loader for pair in batch]).cpu().numpy()

    train_flags_t = [pair.flag_t for batch in train_loader for pair in batch]
    train_flags_v = [pair.flag_v for batch in train_loader for pair in batch]
    classifier_t = LogisticRegression(C=C, random_state=seed)
    classifier_v = LogisticRegression(C=C, random_state=seed)
    classifier_t.fit(x_train, train_flags_t)
    classifier_v.fit(x_train, train_flags_v)

    pred_flags_t = classifier_t.predict(x_test).tolist()
    pred_flags_v = classifier_v.predict(x_test).tolist()
    pred_flags_tv = [pred_flag_t * 2 + pred_flag_v for pred_flag_t, pred_flag_v in zip(pred_flags_t, pred_flags_v)]
    pred_flags_dict = {'t': pred_flags_t, 'v': pred_flags_v, 'tv': pred_flags_tv}

    true_flags_t = [pair.flag_t for batch in test_loader for pair in batch]
    true_flags_v = [pair.flag_v for batch in test_loader for pair in batch]
    true_flags_tv = [pair.flag_tv for batch in test_loader for pair in batch]
    true_flags_dict = {'t': true_flags_t, 'v': true_flags_v, 'tv': true_flags_tv}

    f1_dict = {}
    for task in ('t', 'v', 'tv'):
        f1_dict[task] = f1_score(true_flags_dict[task], pred_flags_dict[task], average='weighted')

    return f1_dict, pred_flags_dict


def linear_probe_quad(train_loader, test_loader, model, C=1.0, seed=0):
    encode(train_loader, model)
    encode(test_loader, model)

    x_train = torch.stack([pair.embedding for batch in train_loader for pair in batch]).cpu().numpy()
    x_test = torch.stack([pair.embedding for batch in test_loader for pair in batch]).cpu().numpy()

    train_flags = [pair.flag_tv for batch in train_loader for pair in batch]
    classifier = LogisticRegression(C=C, random_state=seed)
    classifier.fit(x_train, train_flags)

    true_flags = [pair.flag_tv for batch in test_loader for pair in batch]
    pred_flags = classifier.predict(x_test).tolist()

    return compute_score(true_flags, pred_flags)


def save_results(file_name, results):
    with open(file_name, 'w') as f:
        json.dump(results, f, indent=4)
