import os
from tqdm import tqdm
from typing import List
from pathlib import Path

import cv2
import numpy as np
import torch

import pytorch_grad_cam
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.ablation_layer import AblationLayerVit

from data.loader import load_dataset_bb
from model.model_tv import MyTvModel
from utils import get_parser


parser = get_parser()
parser.add_argument('--state_dict', type=str, default=None)
parser.add_argument('--y', type=int, default=-1, choices=(None, -1, 0, 1, 2, 3))
parser.add_argument('--eigen_smooth', action='store_true', default=True)
parser.add_argument('--method', type=str, default='GradCAM', choices=('GradCAM', ))
parser.add_argument('--use-cuda', action='store_true', default=True, help='Use NVIDIA GPU acceleration')
args = parser.parse_args()
args.raw = True


def forward(cam, pair, targets: List[torch.nn.Module], eigen_smooth: bool = False) -> np.ndarray:
    if cam.compute_input_gradient:
        pair.image.data = torch.autograd.Variable(pair.image.data, requires_grad=True)

    outputs = cam.activations_and_grads(pair)
    if targets is None:
        target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
        targets = [ClassifierOutputTarget(category) for category in target_categories]

    if cam.uses_gradients:
        cam.model.zero_grad()
        loss = sum([target(output) for target, output in zip(targets, outputs)])
        loss.backward(retain_graph=True)

    input_tensor = pair.image.data
    cam_per_layer = cam.compute_cam_per_layer(input_tensor, targets, eigen_smooth)
    return cam.aggregate_multi_layers(cam_per_layer)


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension, like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


if __name__ == '__main__':
    result_path = f'cam/{args.state_dict}'
    os.makedirs(result_path, exist_ok=True)

    dataset_path = Path('/home/resource/datasets/relationship')
    train_set, test_set = load_dataset_bb(dataset_path)
    dataset = train_set + test_set

    print(f'loading pre-trained model {args.state_dict}.pt')
    model = MyTvModel.from_pretrained(args)
    state_dict_path = Path('state_dict')
    state_dict = torch.load(state_dict_path / f'{args.state_dict}.pt')
    model.load_state_dict(state_dict)
    model.eval()
    target_layers = [model.encoder_v.encoder.layer[-1].layernorm_before]

    cam_method = getattr(pytorch_grad_cam, args.method)
    cam = cam_method(model, target_layers, args.use_cuda, reshape_transform)

    # for i, pair in enumerate(tqdm(dataset)):
    tweet_ids = [4246, 1949, 3627, 4424]
    for tweet_id in tqdm(tweet_ids):
        pair = dataset[tweet_id]
        path_to_image = str(dataset.path / f'images/{pair.image.name}.jpg')
        rgb_img = cv2.imread(path_to_image)[:, :, ::-1]
        rgb_img = cv2.resize(rgb_img, (224, 224))
        rgb_img = np.float32(rgb_img) / 255

        targets = [ClassifierOutputTarget(pair.flag_tv if args.y == -1 else args.y)]
        grayscale_cam = forward(cam, pair, targets=targets, eigen_smooth=args.eigen_smooth)

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]

        cam_img = show_cam_on_image(rgb_img, grayscale_cam)
        cv2.imwrite(f'{result_path}/{tweet_id}.jpg', cam_img)

        pair.clean()
        torch.cuda.empty_cache()
