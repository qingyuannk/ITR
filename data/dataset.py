from typing import Optional, List, Dict
from pathlib import Path
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class MyDataPoint:
    def __init__(self):
        self.embedding: Optional[Tensor] = None

    def clean(self):
        del self.embedding
        self.embedding = None


class MyText(MyDataPoint):
    def __init__(self, data: str):
        super().__init__()
        self.data: str = data


class MyImage(MyDataPoint):
    def __init__(self, name: str):
        super().__init__()
        self.name: str = name
        self.data: Optional[Tensor] = None


class MyPair(MyDataPoint):
    def __init__(self, text: MyText, image: MyImage, flag_t: int = -1, flag_v: int = -1):
        super().__init__()
        self.text: MyText = text
        self.image: MyImage = image
        self.flag_t: int = flag_t
        self.flag_v: int = flag_v
        self.flag_tv: int = flag_t * 2 + flag_v

        self.ocr_text: Optional[MyText] = None
        self.cap_text: Optional[MyText] = None

    def clean(self):
        self.text.clean()
        self.image.clean()
        if self.ocr_text:
            self.ocr_text.clean()
        if self.cap_text:
            self.cap_text.clean()
        super().clean()


class MyDataset(Dataset):
    def __init__(self, pairs: List[MyPair], path: Path, roi_data: List = None):
        self.pairs = pairs
        self.path = path
        self.id2roi = {}
        if roi_data:
            for img_datum in roi_data:
                img_id = img_datum['img_id']
                self.id2roi[img_id] = img_datum

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index: int):
        pair = self.pairs[index]
        image = pair.image
        if image.data is None:
            image_path = self.path / image.name
            image.data = Image.open(image_path)
            image.data = image.data.convert('RGB')
        if self.id2roi:
            img_id = image.name.split('.')[0]
            img_info = self.id2roi[img_id]
            image.feats = img_info['features'].copy()
            image.boxes = img_info['boxes'].copy()
            # Normalize the boxes (to 0 ~ 1)
            img_h, img_w = img_info['img_h'], img_info['img_w']
            image.boxes[:, (0, 2)] /= img_w
            image.boxes[:, (1, 3)] /= img_h

        return pair

    def __add__(self, other):
        return MyDataset(self.pairs + other.pairs, self.path, self.load_roi)
