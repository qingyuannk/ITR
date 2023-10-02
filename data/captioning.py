import torch
import numpy as np
from fairseq import utils, tasks
from fairseq import checkpoint_utils
from utils.eval_utils import eval_step
from tasks.mm_tasks.caption import CaptionTask
from models.ofa import OFAModel
from PIL import Image
from torchvision import transforms

import os
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--length', type=int, default=20000)
args = parser.parse_args()

# Register caption task
tasks.register_task('caption', CaptionTask)

# Turn on cuda if GPU is available
use_cuda = torch.cuda.is_available()
# Use fp16 only when GPU is available
use_fp16 = False

# Load pretrained ckpt & config
overrides = {"bpe_dir": "utils/BPE", "eval_cider": False, "beam": 5, "max_len_b": 16, "no_repeat_ngram_size": 3,
             "seed": 7}
models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
    utils.split_paths('checkpoints/caption.pt'),
    arg_overrides=overrides
)

# Move models to GPU
for model in models:
    model.eval()
    if use_fp16:
        model.half()
    if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
        print('********** use GPU **********')
        model.cuda()
    else:
        print('********** use CPU **********')
    model.prepare_for_inference_(cfg)

# Initialize generator
generator = task.build_generator(models, cfg.generation)

# Image transform
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

patch_resize_transform = transforms.Compose([
    lambda image: image.convert("RGB"),
    transforms.Resize((cfg.task.patch_image_size, cfg.task.patch_image_size),
                      interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

# Text preprocess
bos_item = torch.LongTensor([task.src_dict.bos()])
eos_item = torch.LongTensor([task.src_dict.eos()])
pad_idx = task.src_dict.pad()


def encode_text(text, length=None, append_bos=False, append_eos=False):
    s = task.tgt_dict.encode_line(
        line=task.bpe.encode(text),
        add_if_not_exist=False,
        append_eos=False
    ).long()
    if length is not None:
        s = s[:length]
    if append_bos:
        s = torch.cat([bos_item, s])
    if append_eos:
        s = torch.cat([s, eos_item])
    return s


# Construct input for caption task
def construct_sample(image: Image):
    patch_image = patch_resize_transform(image).unsqueeze(0)
    patch_mask = torch.tensor([True])
    src_text = encode_text(" what does the image describe?", append_bos=True, append_eos=True).unsqueeze(0)
    src_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in src_text])
    sample = {
        "id": np.array(['42']),
        "net_input": {
            "src_tokens": src_text,
            "src_lengths": src_length,
            "patch_images": patch_image,
            "patch_masks": patch_mask
        }
    }
    return sample


# Function to turn FP32 to FP16
def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t


twitter100k_path = Path('/home/resource/datasets/twitter100k')
relationship_path = Path('/home/resource/datasets/relationship')


def main(path):
    images_path = path / 'images'
    assert images_path.exists()

    os.makedirs(path / 'ofa', exist_ok=True)
    caption_path = path / 'ofa'
    assert caption_path.exists()

    img_file_name_list = sorted([img_file_name for img_file_name in os.listdir(images_path)],
                                key=lambda x: int(x.split('.')[0]))
    img_file_name_list = img_file_name_list[args.start:args.start + args.length]
    for img_file_name in tqdm(img_file_name_list):
        img_path = images_path / img_file_name
        assert img_path.exists()

        image = Image.open(img_path)
        # Construct input sample & preprocess for GPU if cuda available
        sample = construct_sample(image)
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample

        # Run eval step for caption
        with torch.no_grad():
            result, scores = eval_step(task, generator, models, sample)

        caption = result[0]['caption']

        img_id = img_file_name.split('.')[0]
        txt_file_name = f'{img_id}.txt'
        txt_path = caption_path / txt_file_name
        with open(txt_path, 'w') as txt_file:
            txt_file.writelines(caption)


if __name__ == '__main__':
    main(twitter100k_path)
    # main(relationship_path)
