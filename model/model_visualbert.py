import torch
from typing import List
from transformers import VisualBertConfig, BertTokenizer, VisualBertModel
from data.dataset import MyText, MyPair
from .model_singletower import MySingleTowerModel


class MyVisualBertModel(MySingleTowerModel):
    def __init__(self, args):
        device = torch.device(f'cuda:{args.cuda}')
        encoder_path = f'/data1/modes/transformers/{args.encoder_m}'
        config = VisualBertConfig.from_pretrained(encoder_path)
        tokenizer = BertTokenizer.from_pretrained('/data1/modes/transformers/bert-base-uncased')
        if args.raw:
            encoder = VisualBertModel.from_config(config)
        else:
            encoder = VisualBertModel.from_pretrained(encoder_path)

        super().__init__(
            device=device,
            config=config,
            processor_t=tokenizer,
            encoder=encoder,
            k=args.k,
            dropout=args.dropout,
        )

    def encode_t(self, texts: List[MyText]):
        if self.use_cache(self.encoder, texts): return

        for batch in self.split(texts):
            inputs = [text.data for text in batch]
            inputs = self.processor_t(inputs, padding=True, truncation=True, return_tensors='pt',
                                      max_length=self.config.max_position_embeddings).to(self.device)
            lengths = torch.sum(inputs['attention_mask'], dim=1)

            outputs = self.encoder(**inputs, return_dict=True)
            for text, length, hidden_state in zip(batch, lengths, outputs.last_hidden_state):
                text.embedding = torch.mean(hidden_state[:length], dim=0)

        return torch.stack([text.embedding for text in texts])

    def encode_m(self, pairs: List[MyPair]):
        if self.use_cache(self.encoder, pairs): return

        for batch in self.split(pairs):
            text = [pair.text.data for pair in batch]
            inputs = self.processor_t(text=text, padding=True, truncation=True, return_tensors='pt',
                                      max_length=self.config.max_position_embeddings).to(self.device)
            lengths = torch.sum(inputs['attention_mask'], dim=1)

            visual_embeds = [torch.tensor(pair.image.feats) for pair in batch]
            visual_embeds = torch.stack(visual_embeds).to(self.device)
            image_length = visual_embeds.shape[1]
            inputs["visual_embeds"] = visual_embeds

            outputs = self.encoder(**inputs)
            for pair, length, hidden_state in zip(batch, lengths, outputs.last_hidden_state):
                pair.text.embedding = torch.mean(hidden_state[:length], dim=0)
            for pair, hidden_state in zip(batch, outputs.last_hidden_state):
                pair.image.embedding = torch.mean(hidden_state[-image_length:], dim=0)
            for pair, embedding in zip(batch, outputs.pooler_output):
                pair.embedding = embedding

    def encode(self, pairs: List[MyPair], fuse: str = 't+v'):
        if fuse == 't+v':
            self.encode_m(pairs)
        else:
            if 't' in fuse:
                texts = [pair.text for pair in pairs]
                self.encode_t(texts)
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
