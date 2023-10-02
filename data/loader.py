import sys
import csv
import base64
import time
import logging
import re
import numpy as np
from pathlib import Path
from typing import List
from .dataset import MyText, MyImage, MyPair, MyDataset


csv.field_size_limit(sys.maxsize)

TWEET_LENGTH_LIMIT = 140
SPLIT = 3576
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]


def load_obj_tsv(fname, topk=None):
    """ Load object features from tsv file.

    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """

    data = []
    start_time = time.time()
    logging.info("Start to load Faster-RCNN detected objects from %s" % fname)

    with open(fname) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for i, item in enumerate(reader):

            for key in ['img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])

            boxes = item['num_boxes']
            decode_config = [
                ('objects_id', (boxes,), np.int64),
                ('objects_conf', (boxes,), np.float32),
                ('attrs_id', (boxes,), np.int64),
                ('attrs_conf', (boxes,), np.float32),
                ('boxes', (boxes, 4), np.float32),
                ('features', (boxes, -1), np.float32),
            ]
            for key, shape, dtype in decode_config:
                item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                item[key] = item[key].reshape(shape)

            data.append(item)
            if topk is not None and len(data) == topk:
                break

    elapsed_time = time.time() - start_time
    logging.info("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))
    return data


def normalize_text(text: str):
    # remove the ending URL which is not part of the text
    url_re = r' http[s]?://t.co/\w+$'
    text = re.sub(url_re, '', text)
    return text


def read_text(path: Path, pairs: List[MyPair], truncate: bool = True):
    for i, pair in enumerate(pairs):
        tweet_id = pair.image.name.split('.')[0]
        file_name = f'{tweet_id}.txt'

        with open(path/'ocr'/file_name) as ocr_file:
            text = ocr_file.read()
            if truncate: text = text[:TWEET_LENGTH_LIMIT]  # tweet length limit
            pair.ocr_text = MyText(text)

        with open(path/'ofa'/file_name) as cap_file:
            text = cap_file.read()
            if truncate: text = text[:TWEET_LENGTH_LIMIT]  # tweet length limit
            pair.cap_text = MyText(text)


def load_dataset_100k(path: Path, size: int = 100000, load_roi: bool = False):
    assert path.exists()

    with open(path/'text.txt', encoding='utf-8') as txt_file:
        pairs = [MyPair(MyText(text[:TWEET_LENGTH_LIMIT]), MyImage(f'{i+1}.jpg')) for i, text in enumerate(txt_file)]

    read_text(path, pairs)

    roi_data = None
    if load_roi:
        roi_data = load_obj_tsv(path/'butd.tsv')

    return MyDataset(pairs[:size], path/'images', roi_data)


def load_dataset_bb(path: Path, split: int = SPLIT, normalize: bool = True, load_roi: bool = False):
    assert path.exists()

    with open(path/'data.csv', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file, doublequote=False, escapechar='\\')
        pairs = [MyPair(
            text=MyText(normalize_text(row['tweet']) if normalize else row['tweet']),
            image=MyImage(f"T{row['tweet_id']}.jpg"),
            flag_t=int(row['text_is_represented']),
            flag_v=int(row['image_adds'])
        ) for row in csv_reader]

    roi_data = None
    if load_roi:
        roi_data = load_obj_tsv(path/'butd.tsv')

    
    train_f = open("test.csv",'a', encoding = 'utf-8', newline = '')
    # train_writer = csv.DictWriter(train_f, fieldnames=["text", "image", "flag_t", "flag_v", "flag_tv"])
    # train_writer.writeheader()
    # for pair in pairs[split:]:
    #     text=pair.text.data
    #     image = pair.image.name 
    #     flag_t = pair.flag_t
    #     flag_v = pair.flag_v
    #     flag_tv = pair.flag_tv
    #     train_writer.writerow({"text":text, "image":image, "flag_t":flag_t, "flag_v": flag_v, "flag_tv":flag_tv})



    return MyDataset(pairs[:split], path/'images', roi_data), MyDataset(pairs[split:], path/'images', roi_data)


def load_dataset_fakenews(path: Path):
    assert path.exists()

    with open(path/'annotation.csv', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        pairs = [
            MyPair(text=MyText(row['Title']), image=MyImage(f'{i}.jpg'), flag_t=int(row['Label']))
            for i, row in enumerate(csv_reader)
        ]

    return MyDataset(pairs, path/'images')


if __name__ == '__main__':
    itr_path = Path('/data1/datasets/relationship')
    itr_train_set, itr_test_set = load_dataset_bb(itr_path, load_roi=True)
