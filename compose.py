import os
from pathlib import Path
from tqdm import tqdm

settings = ['fine-tune(vanilla)_1|1_0']\
           + [f'fine-tune(bertweet-base+vit-base-patch16-224-in21k({i})_5k_100k_0_1)_1|1_0' for i in [2, 3, 4, 5]]
           # + [f'fine-tune(bertweet-base+vit-base-patch16-224-in21k({i}+5)_5k_100k_0_1)_1|1_0' for i in [2, 3, 4]]

tweet_ids = list(range(4471))

path_att = Path('att')
path_cam = Path('cam')
path_result = Path('case')

os.makedirs(path_result, exist_ok=True)

for i, setting in enumerate(settings):
    # for tweet_id in tqdm(tweet_ids):
    tweet_ids = [4246, 1949, 3627, 4424]
    for tweet_id in tqdm(tweet_ids):
        path_img_att = path_att/setting/f'{tweet_id}.pdf'
        path_img_cam = path_cam/setting/f'{tweet_id}.jpg'
        assert path_img_att.exists()
        assert path_img_cam.exists()
        os.system(f'cp "{path_img_att}" "{path_result}/att_{tweet_id}_{i}.pdf"')
        os.system(f'cp "{path_img_cam}" "{path_result}/cam_{tweet_id}_{i}.jpg"')
