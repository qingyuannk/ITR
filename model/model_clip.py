import torch
from transformers import CLIPConfig, CLIPProcessor, CLIPModel
from .model_twotower import MyTwoTowerModel


class MyClipModel(MyTwoTowerModel):
    def __init__(self, args):
        device = torch.device(f'cuda:{args.cuda}')
        encoder_m_path = f'/data1/modes/transformers/{args.encoder_m}'
        config_m = CLIPConfig.from_pretrained(encoder_m_path)
        processor_m = CLIPProcessor.from_pretrained(encoder_m_path)
        if args.raw:
            encoder_m = CLIPModel.from_config(config_m)
        else:
            encoder_m = CLIPModel.from_pretrained(encoder_m_path)

        super().__init__(
            device=device,
            processor_t=processor_m.tokenizer,
            processor_v=processor_m.feature_extractor,
            encoder_t=encoder_m.text_model,
            encoder_v=encoder_m.vision_model,
            config_t=config_m.text_config,
            config_v=config_m.vision_config,
            proj_t=encoder_m.text_projection,
            proj_v=encoder_m.visual_projection,
            freeze_t=args.freeze_t,
            freeze_v=args.freeze_v,
            k=args.k,
            dropout=args.dropout,
        )
