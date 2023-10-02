from typing import List
import torch
from torch import nn
from transformers import PreTrainedModel, CLIPConfig, CLIPProcessor, CLIPModel
from data.dataset import MyDataPoint, MyPair


def split(data_points, batch_size=2):
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
        processor_m,
        encoder_m: PreTrainedModel,
        hid_dim_m: int,
        dropout: float = 0.0,
    ):
        super(MyModel, self).__init__()
        self.device = device
        self.processor_m = processor_m
        self.encoder_m = encoder_m
        self.dropout = nn.Dropout(dropout)

        self.head1 = nn.Linear(hid_dim_m + hid_dim_m, 2)
        self.head2 = nn.Linear(hid_dim_m + hid_dim_m, 4)
        self.head3 = nn.Linear(hid_dim_m + hid_dim_m, 4)
        self.head4 = nn.Linear(hid_dim_m + hid_dim_m, 4)
        self.proj_t = nn.Linear(hid_dim_m, 512, bias=False)
        self.proj_v = nn.Linear(hid_dim_m, 512, bias=False)
        self.classifier = nn.Linear(hid_dim_m + hid_dim_m, 4)

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
        encoder_m_path = f'{models_path}/transformers/{args.encoder_m}'
        config_m = CLIPConfig.from_pretrained(encoder_m_path)
        processor_m = CLIPProcessor.from_pretrained(encoder_m_path)
        encoder_m = CLIPModel.from_pretrained(encoder_m_path)

        return cls(
            device=device,
            processor_m=processor_m,
            encoder_m=encoder_m,
            hid_dim_m=config_m.projection_dim,
            dropout=args.dropout,
        )

    def encode(self, pairs: List[MyPair], fuse: str = 't+v'):
        text = [pair.text.data for pair in pairs]
        image = [pair.image.data for pair in pairs]
        inputs = self.processor_m(text=text, images=image, return_tensors="pt", padding=True).to(self.device)

        outputs = self.encoder_m(**inputs)
        text_embeds = outputs.text_embeds
        image_embeds = outputs.image_embeds

        fuse_func = self.fuse_func_map[fuse]
        for pair, text_embed, image_embed in zip(pairs, text_embeds, image_embeds):
            pair.text.embedding = text_embed
            pair.image.embedding = image_embed
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
