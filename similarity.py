from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from data.loader import load_dataset_100k, load_dataset_bb
from model.model_tv import MyTvModel
from utils import get_parser, seed_everything, save_results
from utils import encode, cluster, train, evaluate_cluster


def main():
    parser = get_parser()
    parser.add_argument('--k', type=int, default=4, choices=(2, 4))
    parser.add_argument('--size', type=int, default=100000)
    parser.add_argument('--subsize', type=int, default=5000)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--preprocess', type=str, default=None, choices=(None, 'normalize', 'scale'))
    parser.add_argument('--task_ids', type=str, default=None, choices=('2+5', '3+5', '4+5'))
    args = parser.parse_args()
    print(args)

    model_id = f'{args.encoder_t}+{args.encoder_v}' \
               f'({args.task_ids + ("raw" if args.raw else "")})_' \
               f'{int(args.subsize / 1000)}k_{int(args.size / 1000)}k_{args.seed}'

    seed_everything(args.seed)

    twitter100k_path = Path('/home/resource/datasets/twitter100k')
    relationship_path = Path('/home/resource/datasets/relationship')
    train_set = load_dataset_100k(twitter100k_path, args.size)
    dev_set, _ = load_dataset_bb(relationship_path)
    dev_loader = DataLoader(dev_set, batch_size=args.bs, collate_fn=list)

    num_iter = args.size // args.subsize
    indices_list = [list(range(i * args.subsize, (i + 1) * args.subsize)) for i in range(num_iter)]
    split = int(num_iter * 0.8)
    indices_list_train, indices_list_test = indices_list[:split], indices_list[split:]

    indices_test = []
    for indices in indices_list_test:
        indices_test += indices
    subtrain_set = Subset(train_set, indices_test)
    subtrain_loader = DataLoader(subtrain_set, batch_size=args.bs, collate_fn=list)

    model = MyTvModel.from_pretrained(args)
    optimizer = getattr(torch.optim, args.optim)(model.parameters(), args.lr)

    fuse_map = {'2': 't+v', '3': 't+o', '4': 't+c'}
    cluster_id = None
    for task_id in args.task_ids:
        if task_id in fuse_map:
            cluster_id = task_id

    scores_list, centers_list, map_list = [], [], []
    for epoch in range(args.epochs):
        for i, indices in enumerate(indices_list):
            subset = Subset(train_set, indices)
            train_loader = DataLoader(subset, batch_size=args.bs, collate_fn=list)

            encode(tqdm(train_loader), model, fuse_map[cluster_id])
            kmeans = cluster(subset, args.k, cluster_id, args.seed, args.preprocess)
            centers_list.append(kmeans.cluster_centers_.tolist())

            head = torch.nn.Linear(model.hid_dim_t + model.hid_dim_v, args.k).to(model.device)
            setattr(model, f'head{cluster_id}', head)
            model.classifier = head

            iteration = epoch * num_iter + i + 1
            loss = train(tqdm(train_loader), model, F.cross_entropy, optimizer, args.task_ids)
            print(f"iteration #{iteration:02d}, loss: {loss:.2f}")

            encode(tqdm(subtrain_loader), model, fuse_map[cluster_id])
            x = torch.stack([pair.embedding for pair in subtrain_set]).cpu().numpy()
            pseudo_flags = kmeans.predict(x)

            encode(tqdm(dev_loader), model)
            f1_dict, flags_dict, flag_map = evaluate_cluster(kmeans, dev_set)
            map_list.append(flag_map)

            scores = []
            for c in range(args.k):
                pairs = [pair for i, pair in enumerate(subtrain_set) if pseudo_flags[i] == c]
                embeddings_t = torch.stack([pair.text.embedding for pair in pairs])
                embeddings_v = torch.stack([pair.image.embedding for pair in pairs])
                embeddings_t = model.proj_t(embeddings_t)
                embeddings_v = model.proj_v(embeddings_v)
                scores.append(torch.mean(F.cosine_similarity(embeddings_t, embeddings_v)).item())

            scores_list.append(scores)

            for pair in subset:
                pair.clean()
            for pair in subtrain_set:
                pair.clean()
            torch.cuda.empty_cache()

            results_similarity = {
                'config': vars(args),
                'scores_list': scores_list,
                'map_list': map_list,
                'centers_list': centers_list,
            }
            save_results(f'log/similarity_{model_id}.json', results_similarity)


if __name__ == '__main__':
    main()
