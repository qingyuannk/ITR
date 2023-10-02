import torch
from transformers import ViltConfig, ViltProcessor, ViltModel
from .model_singletower import MySingleTowerModel


class MyViltModel(MySingleTowerModel):
    def __init__(self, args):
        device = torch.device(f'cuda:{args.cuda}')
        encoder_path = f'/data1/modes/transformers/{args.encoder_m}'
        config = ViltConfig.from_pretrained(encoder_path)
        processor_m = ViltProcessor.from_pretrained(encoder_path)
        if args.raw:
            encoder = ViltModel.from_config(config)
        else:
            encoder = ViltModel.from_pretrained(encoder_path)

        super().__init__(
            device=device,
            config=config,
            processor_m=processor_m,
            encoder=encoder,
            k=args.k,
            dropout=args.dropout,
        )

    def encode(self, pairs, fuse: str = ''):
        for batch in self.split(pairs):
            text = [pair.text.data for pair in batch]
            images = [pair.image.data for pair in batch]
            inputs = self.processor_m(text=text, images=images, padding=True, truncation=True, return_tensors='pt')
            outputs = self.encoder(**inputs, return_dict=True)
            for pair, embedding in zip(batch, outputs.pooler_output):
                pair.embedding = embedding

        return torch.stack([pair.embedding for pair in pairs])
