# Multimodal Pre-trained Framework for Aligning Image-Text Relation Semantics

## Environment
* python
* numpy
* torch
* torchvision
* transformers
**(pip install -r requirements.txt)**

## Data
Please refer [this repository](https://github.com/danielpreotiuc/text-image-relationship) for the text-image relationship dataset and [this repository](https://github.com/huyt16/Twitter100k) for the Twitter100k dataset.

## Pretrained Models/Embeddings
Download pretrained BERTWEET-Base from [here](https://huggingface.co/bert-base-uncased/tree/main) and put it in [this directory](resources/transformers).

Download pretrained ViT from [here](https://download.pytorch.org/models/resnet152-394f9c45.pth), rename the binary file as "resnet101.pth" and put it in [this directory](resources/cnn).

Download pretrained Twitter Word Embedding from [here](https://flair.informatik.hu-berlin.de/resources/embeddings/token/twitter.gensim.vectors.npy) and put it in [this directory](resources/embeddings).

Download our pretrained ours from [here](https://drive.google.com/file/d/1LYdAN8nje18frwa3ZJrC-JQxrh_COrgg/view?usp=share_link) and put it in [this directory](resources/pretrain/models).

## Usage
### Pretrain
`python pretrain.py --cuda [GPU ID] --encoder [encoder name] --task_ids [task IDs] (--ocr)`

### Linear probe
`python linear_probe.py --cuda [GPU ID] --encoder [encoder name] --task_ids [task IDs] (--ocr)`

### Finetune
`python finetune.py --cuda [GPU ID] --encoder [encoder name] --task_ids [task IDs] (--ocr)`

### Result Analysis
#### statistic result
`python statistic.py --eval [evaluation setting, e.g. fine-tune] --encoder [encoder name]`
#### visualize for gradcam
`python gradcam.py --encoder [encoder name]`


