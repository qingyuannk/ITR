from typing import List
import torch
from torch import nn
from data.dataset import MyText, MyImage, MyPair
from .model_base import MyBaseModel


class MyTwoTowerModel(MyBaseModel):
    def __init__(
        self,
        device,
        processor_t,
        processor_v,
        encoder_t,
        encoder_v,
        freeze_t,
        freeze_v,
        k,
        proj_dim=512,
        dropout=0.0,
        config_t=None,
        config_v=None,
        proj_t=None,
        proj_v=None,
    ):
        super().__init__()
        self.device = device
        self.processor_t = processor_t
        self.processor_v = processor_v
        self.encoder_t = encoder_t
        self.encoder_v = encoder_v
        self.config_t = config_t
        self.config_v = config_v
        self.dropout = nn.Dropout(dropout)

        hid_dim_t = config_t.hidden_size
        hid_dim_v = config_v.hidden_size
        self.head1 = nn.Linear(hid_dim_t + hid_dim_v, k)
        self.head2 = nn.Linear(hid_dim_t + hid_dim_v, k)
        self.head3 = nn.Linear(hid_dim_t + hid_dim_v, k)
        self.proj_t = proj_t if proj_t else nn.Linear(hid_dim_t, proj_dim, bias=False)
        self.proj_v = proj_v if proj_v else nn.Linear(hid_dim_v, proj_dim, bias=False)
        self.classifier = nn.Linear(hid_dim_t + hid_dim_v, k)

        self.fuse_func_map = {
            't+v': lambda pair: torch.cat((pair.text.embedding, pair.image.embedding), dim=0),
            't+o': lambda pair: torch.cat((pair.text.embedding, pair.ocr_text.embedding), dim=0),
            't+c': lambda pair: torch.cat((pair.text.embedding, pair.cap_text.embedding), dim=0),
        }

        if freeze_t:
            for parameter in self.encoder_t.parameters():
                parameter.requires_grad = False
        if freeze_v:
            for parameter in self.encoder_v.parameters():
                parameter.requires_grad = False

        self.to(device)

    def encode_t(self, texts: List[MyText]):
        if self.use_cache(self.encoder_t, texts): return

        for batch in self.split(texts):
            inputs = [text.data for text in batch]
            inputs = self.processor_t(inputs, padding=True, truncation=True, return_tensors='pt',
                                      max_length=self.config_t.max_position_embeddings).to(self.device)
            outputs = self.encoder_t(**inputs, return_dict=True)
            embeddings = outputs.pooler_output
            for text, embedding in zip(batch, embeddings):
                text.embedding = embedding

        return torch.stack([text.embedding for text in texts])

    def encode_v(self, images: List[MyImage]):
        if self.use_cache(self.encoder_v, images): return

        for batch in self.split(images):
            inputs = [image.data for image in batch]
            inputs = self.processor_v(inputs, return_tensors='pt').to(self.device)
            outputs = self.encoder_v(**inputs, return_dict=True)
            embeddings = outputs.pooler_output
            for image, embedding in zip(batch, embeddings):
                image.embedding = embedding.squeeze()

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
