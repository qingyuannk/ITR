import os
from pathlib import Path
from tqdm import tqdm
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en')

def main(path):
    images_path = path/'images'
    assert images_path.exists()

    os.makedirs(path/'ocr', exist_ok=True)
    ocr_path = path/'ocr'
    assert ocr_path.exists()

    for img_file_name in tqdm(os.listdir(images_path)):
        img_path = images_path/img_file_name
        assert img_path.exists()

        result = ocr.ocr(str(img_path), cls=True)

        boxes = [line[0] for line in result]
        texts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]

        img_id = img_file_name.split('.')[0]
        txt_file_name = f'{img_id}.txt'
        txt_path = ocr_path/txt_file_name
        with open(txt_path, 'w') as txt_file:
            txt_file.writelines(texts)


if __name__ == '__main__':
    twitter100k_path = Path('/home/resource/datasets/twitter100k')
    relationship_path = Path('/home/resource/datasets/relationship')
    # main(twitter100k_path)
    main(relationship_path)