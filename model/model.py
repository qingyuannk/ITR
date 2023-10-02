from typing import List
import torch
from torch import nn
import torchvision
from transformers import PreTrainedTokenizer, PreTrainedModel, AutoConfig, AutoTokenizer, AutoModel
from data.dataset import MyDataPoint, MyText, MyImage, MyPair


def split(data_points, batch_size = 2):
    num_batch = (len(data_points) + batch_size - 1) // batch_size
    return [data_points[i * batch_size:(i + 1) * batch_size] for i in range(num_batch)]


def use_cache(module: nn.Module, data_points: List[MyDataPoint]):
    for parameter in module.parameters():
        if parameter.requires_grad:
            return False
    for data_point in data_points:
        if data_point.embedding is None:
            return False
    return True


class MyModel(nn.Module):
    def __init__(
        self,
        device: torch.device,
        tokenizer: PreTrainedTokenizer,
        encoder_t: PreTrainedModel,
        encoder_v: nn.Module,
        hid_dim_t: int,
        hid_dim_v: int,
        pool_t: str,
        pool_v: str,
        dropout: float = 0.0,
    ):
        super(MyModel, self).__init__()
        self.device = device
        self.tokenizer = tokenizer
        self.encoder_t = encoder_t
        self.encoder_v = encoder_v
        self.hid_dim_t = hid_dim_t
        self.hid_dim_v = hid_dim_v
        self.pool_t = pool_t
        self.pool_v = pool_v
        self.dropout = nn.Dropout(dropout)

        self.head1 = nn.Linear(hid_dim_t + hid_dim_v, 2)
        self.head2 = nn.Linear(hid_dim_t + hid_dim_v, 4)
        self.head3 = nn.Linear(hid_dim_t + hid_dim_v, 4)
        self.head4 = nn.Linear(hid_dim_t + hid_dim_v, 4)
        self.proj_t = nn.Linear(hid_dim_t, 512, bias=False)
        self.proj_v = nn.Linear(hid_dim_v, 512, bias=False)
        self.classifier = nn.Linear(hid_dim_t + hid_dim_v, 4)

        self.fuse_func_map = {
            't+v': lambda pair: torch.cat((pair.text.embedding, pair.image.embedding), dim=0),
            't+o': lambda pair: torch.cat((pair.text.embedding, pair.ocr_text.embedding), dim=0),
            't+c': lambda pair: torch.cat((pair.text.embedding, pair.cap_text.embedding), dim=0),
        }

        self.to(device)

    @classmethod
    def from_pretrained(cls, args):
        device = torch.device(f'cuda:{args.cuda}')
        models_path = '/data1/models'

        encoder_t_path = f'{models_path}/transformers/{args.encoder_t}'
        if 'bertweet' in args.encoder_t:
            tokenizer = AutoTokenizer.from_pretrained(encoder_t_path, use_fast=False)
        else:
            tokenizer = AutoTokenizer.from_pretrained(encoder_t_path)
        if not args.raw:
            encoder_t = AutoModel.from_pretrained(encoder_t_path)
        else:
            config_t = AutoConfig.from_pretrained(encoder_t_path)
            encoder_t = AutoModel.from_config(config_t)
        if args.freeze_t:
            for parameter in encoder_t.parameters():
                parameter.requires_grad = False
        hid_dim_t = {
            'bert-base-uncased': 768,
            'roberta-base': 768,
            'bertweet-base': 768,
        }[args.encoder_t]

        if 'vit' in args.encoder_v:
            encoder_v_path = f'{models_path}/transformers/{args.encoder_v}'
            if not args.raw:
                encoder_v = AutoModel.from_pretrained(encoder_v_path)
            else:
                config_v = AutoConfig.from_pretrained(encoder_v_path)
                encoder_v = AutoModel.from_config(config_v)
        else:
            encoder_v = getattr(torchvision.models, args.encoder_v)()
            if not args.raw:
                encoder_v.load_state_dict(torch.load(f'{models_path}/cnn/{args.encoder_v}.pth'))
            if 'efficientnet' in args.encoder_v:
                encoder_v.classifier = torch.nn.Identity()
            else: # 'resnet' in args.encoder_v
                encoder_v.fc = torch.nn.Identity()
        if args.freeze_v:
            for parameter in encoder_v.parameters():
                parameter.requires_grad = False

        hid_dim_v = {
            'resnet101': 2048,
            'resnet152': 2048,
            'efficientnet_b4': 1792,
            'vit-base-patch16-224-in21k': 768,
        }[args.encoder_v]

        return cls(
            device=device,
            tokenizer=tokenizer,
            encoder_t=encoder_t,
            encoder_v=encoder_v,
            hid_dim_t=hid_dim_t,
            hid_dim_v=hid_dim_v,
            pool_t=args.pool_t,
            pool_v=args.pool_v,
            dropout=args.dropout,
        )

    def encode_t(self, texts: List[MyText]):
        if use_cache(self.encoder_t, texts): return

        for batch in split(texts):
            sentences = [text.data for text in batch]
            inputs = self.tokenizer(sentences, padding=True, return_tensors='pt').to(self.device)
            outputs = self.encoder_t(**inputs, return_dict=True)
            last_hidden_state = outputs.last_hidden_state

            if self.pool_t == 'cls':
                index_of_cls_token = 0
                embeddings = last_hidden_state[:, index_of_cls_token]
            else: # self.pool_t == 'mean'
                lengths = torch.sum(inputs['attention_mask'], dim=1)
                embeddings = [torch.mean(hidden_state[:length], dim=0)
                              for length, hidden_state in zip(lengths, last_hidden_state)]

            for text, embedding in zip(batch, embeddings):
                text.embedding = embedding

        return torch.stack([text.embedding for text in texts])

    def encode_v(self, images: List[MyImage]):
        if use_cache(self.encoder_v, images): return

        for batch in split(images):
            pixels = torch.stack([image.data for image in batch]).to(self.device)
            embeddings = self.encoder_v(pixels)
            if not isinstance(embeddings, torch.Tensor): # ViT outputs
                if self.pool_v == 'cls':
                    index_of_cls_token = 0
                    embeddings = embeddings.last_hidden_state[:, index_of_cls_token]
                else:  # self.pool_v == 'mean'
                    embeddings = torch.mean(embeddings.last_hidden_state, dim=1)

            for image, embedding in zip(batch, embeddings):
                image.embedding = embedding

            del pixels

        return torch.stack([image.embedding for image in images])

    def encode(self, pairs: List[MyPair], fuse: str = 't+v'):
        if 't' in fuse:
            texts = [pair.text for pair in pairs]
            self.encode_t(texts)

        if 'v' in fuse:
            images = [pair.image for pair in pairs]
            self.encode_v(images)

        if 'o' in fuse:
            texts = [pair.ocr_text for pair in pairs]
            self.encode_t(texts)

        if 'c' in fuse:
            texts = [pair.cap_text for pair in pairs]
            self.encode_t(texts)

        fuse_func = self.fuse_func_map[fuse]
        for pair in pairs:
            pair.embedding = fuse_func(pair)

        return torch.stack([pair.embedding for pair in pairs])

    def forward(self, pairs: List[MyPair]):
        if isinstance(pairs, MyPair):
            pairs = [pairs]
        feats = self.encode(pairs)
        feats = self.dropout(feats)
        logits = self.classifier(feats)
        return logits

    def predict(self, data_loader):
        pred_flags = []
        self.eval()
        with torch.no_grad():
            for pairs in data_loader:
                logits = self.forward(pairs)
                pred_flags += torch.argmax(logits, dim=1).tolist()
        return pred_flags
