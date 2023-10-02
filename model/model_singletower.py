import torch
from torch import nn
from .model_base import MyBaseModel


class MySingleTowerModel(MyBaseModel):
    def __init__(
        self,
        device,
        config,
        encoder,
        k,
        processor_t=None,
        processor_v=None,
        processor_m=None,
        proj_dim=512,
        dropout=0.0,
    ):
        super().__init__()
        self.device = device
        self.config = config
        self.processor_t = processor_t
        self.processor_v = processor_v
        self.processor_m = processor_m
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)

        hid_dim = config.hidden_size
        self.head1 = nn.Linear(hid_dim, k)
        self.head2 = nn.Linear(hid_dim, k)
        self.head3 = nn.Linear(hid_dim, k)
        self.proj_t = nn.Linear(hid_dim, proj_dim, bias=False)
        self.proj_v = nn.Linear(hid_dim, proj_dim, bias=False)
        self.classifier = nn.Linear(hid_dim, k)

        self.fuse_func_map = {
            't+v': lambda pair: torch.cat((pair.text.embedding, pair.image.embedding), dim=0),
            't+o': lambda pair: torch.cat((pair.text.embedding, pair.ocr_text.embedding), dim=0),
            't+c': lambda pair: torch.cat((pair.text.embedding, pair.cap_text.embedding), dim=0),
        }

        self.to(device)
