import torch
from typing import List
from transformers import LxmertConfig, LxmertTokenizer, LxmertModel
from data.dataset import MyPair
from .model_singletower import MySingleTowerModel


class MyLxmertModel(MySingleTowerModel):
    def __init__(self, args):
        device = torch.device(f'cuda:{args.cuda}')
        encoder_path = f'/data1/modes/transformers/{args.encoder_m}'
        config = LxmertConfig.from_pretrained(encoder_path)
        tokenizer = LxmertTokenizer.from_pretrained(encoder_path)
        if args.raw:
            encoder = LxmertModel.from_config(config)
        else:
            encoder = LxmertModel.from_pretrained(encoder_path)

        super().__init__(
            device=device,
            config=config,
            processor_t=tokenizer,
            encoder=encoder,
            k=args.k,
            dropout=args.dropout,
        )

    def encode(self, pairs: List[MyPair], fuse: str = ''):
        if self.use_cache(self.encoder, pairs): return

        for batch in self.split(pairs, batch_size=2):
            text = [pair.text.data for pair in batch]
            inputs = self.processor_t(text=text, padding=True, truncation=True, return_tensors='pt',
                                      max_length=self.config.max_position_embeddings).to(self.device)

            visual_feats = [torch.tensor(pair.image.feats) for pair in batch]
            visual_feats = torch.stack(visual_feats).to(self.device)
            visual_pos = [torch.tensor(pair.image.boxes) for pair in batch]
            visual_pos = torch.stack(visual_pos).to(self.device)
            inputs.update({
                "visual_feats": visual_feats,
                "visual_pos": visual_pos,
            })

            outputs = self.encoder(**inputs)
            lengths = torch.sum(inputs['attention_mask'], dim=1)
            for pair, length, hidden_state in zip(batch, lengths, outputs.language_output):
                pair.text.embedding = torch.mean(hidden_state[:length], dim=0)
            for pair, hidden_state in zip(batch, outputs.vision_output):
                pair.image.embedding = torch.mean(hidden_state, dim=0)
            for pair, embedding in zip(batch, outputs.pooled_output):
                pair.embedding = embedding

            del visual_feats
            del visual_pos

        return torch.stack([pair.embedding for pair in pairs])
