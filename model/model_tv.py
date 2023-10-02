import torch
from transformers import AutoConfig, AutoTokenizer, AutoFeatureExtractor, AutoModel
from .model_twotower import MyTwoTowerModel


class MyTvModel(MyTwoTowerModel):
    def __init__(self, args):
        device = torch.device(f'cuda:{args.cuda}')
        models_path = '/data1/models'

        encoder_t_path = f'{models_path}/transformers/{args.encoder_t}'
        config_t = AutoConfig.from_pretrained(encoder_t_path)
        # for BERTweet tokenizer
        use_fast = 'bertweet' not in args.encoder_t
        processor_t = AutoTokenizer.from_pretrained(encoder_t_path, use_fast=use_fast)
        if args.raw:
            encoder_t = AutoModel.from_config(config_t)
        else:
            encoder_t = AutoModel.from_pretrained(encoder_t_path)

        encoder_v_path = f'{models_path}/transformers/{args.encoder_v}'
        config_v = AutoConfig.from_pretrained(encoder_v_path)
        # for CNN backbone
        if not hasattr(config_v, 'hidden_size') and hasattr(config_v, 'hidden_sizes'):
            config_v.hidden_size = config_v.hidden_sizes[-1]
        processor_v = AutoFeatureExtractor.from_pretrained(encoder_v_path)
        if args.raw:
            encoder_v = AutoModel.from_config(config_v)
        else:
            encoder_v = AutoModel.from_pretrained(encoder_v_path)

        super().__init__(
            device=device,
            processor_t=processor_t,
            processor_v=processor_v,
            encoder_t=encoder_t,
            encoder_v=encoder_v,
            config_t=config_t,
            config_v=config_v,
            freeze_t=args.freeze_t,
            freeze_v=args.freeze_v,
            k=args.k,
            dropout=args.dropout,
        )
