import os
import json
import random
import copy
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor

from utils.image_utils import random_augmentation, crop_img
from utils.degradation_utils import Degradation


class TrainDataset(Dataset):
    def __init__(self, args):
        super(TrainDataset, self).__init__()
        self.args = args

        self.hazy_ids = []
        self.rain_ids = []
        self.low_ids = []
        self.hazy_rain_ids = []
        self.low_rain_ids = []
        self.low_haze_ids = []
        self.low_haze_rain_ids = []

        self.D = Degradation(args)
        self.de_temp = 0
        self.de_type = self.args.de_type
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

        self._init_ids()

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])

        self.toTensor = ToTensor()

        with open('prompts.json', 'r') as f:
            self.prompts = json.load(f)

        self.prompts_key = list(self.prompts.keys())

    def _init_ids(self):
        self._init_haze_ids()
        self._init_rain_ids()
        self._init_low_ids()
        self._init_hr_ids()
        self._init_lh_ids()
        self._init_lr_ids()
        self._init_lhr_ids()


    # haze
    def _init_haze_ids(self):
        # 파일 목록 중 이미지 파일만 필터링
        derain_ids = []
        # name_list = os.listdir(self.args.derain_dir)

        for root, dirs, files in os.walk(self.args.derain_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in self.image_extensions and os.path.basename(root) == 'haze':
                    derain_ids.append(os.path.join(root, file))
        self.hazy_ids = derain_ids
        print(self.hazy_ids)
        self.haze_counter = 0
        self.num_haze = len(self.hazy_ids)

    # rain
    def _init_rain_ids(self):
        # 파일 목록 중 이미지 파일만 필터링
        derain_ids = []
        # name_list = os.listdir(self.args.derain_dir)

        for root, dirs, files in os.walk(self.args.derain_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in self.image_extensions and os.path.basename(root) == 'rain':
                    derain_ids.append(os.path.join(root, file))
        self.rain_ids = derain_ids
        self.rain_counter = 0
        self.num_rain = len(self.rain_ids)

    # low
    def _init_low_ids(self):
        # 파일 목록 중 이미지 파일만 필터링
        derain_ids = []
        # name_list = os.listdir(self.args.derain_dir)

        for root, dirs, files in os.walk(self.args.derain_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in self.image_extensions and os.path.basename(root) == 'low':
                    derain_ids.append(os.path.join(root, file))
        self.low_ids = derain_ids
        self.low_counter = 0
        self.num_low = len(self.low_ids)

    # haze rain
    def _init_hr_ids(self):
        # 파일 목록 중 이미지 파일만 필터링
        derain_ids = []
        # name_list = os.listdir(self.args.derain_dir)

        for root, dirs, files in os.walk(self.args.derain_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in self.image_extensions and os.path.basename(root) == 'haze_rain':
                    derain_ids.append(os.path.join(root, file))
        self.hazy_rain_ids = derain_ids
        self.hr_counter = 0
        self.num_hr = len(self.hazy_rain_ids)

    # low_rain
    def _init_lr_ids(self):
        # 파일 목록 중 이미지 파일만 필터링
        derain_ids = []
        # name_list = os.listdir(self.args.derain_dir)

        for root, dirs, files in os.walk(self.args.derain_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in self.image_extensions and os.path.basename(root) == 'low_rain':
                    derain_ids.append(os.path.join(root, file))
        self.low_rain_ids = derain_ids
        self.lr_counter = 0
        self.num_lr = len(self.low_rain_ids)

    # low haze
    def _init_lh_ids(self):
        # 파일 목록 중 이미지 파일만 필터링
        derain_ids = []
        # name_list = os.listdir(self.args.derain_dir)

        for root, dirs, files in os.walk(self.args.derain_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in self.image_extensions and os.path.basename(root) == 'low_haze':
                    derain_ids.append(os.path.join(root, file))
        self.low_haze_ids = derain_ids
        self.lh_counter = 0
        self.num_lh = len(self.low_haze_ids)

    # low haze rain
    def _init_lhr_ids(self):
        # 파일 목록 중 이미지 파일만 필터링
        derain_ids = []
        # name_list = os.listdir(self.args.derain_dir)

        for root, dirs, files in os.walk(self.args.derain_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in self.image_extensions and os.path.basename(root) == 'low_haze_rain':
                    derain_ids.append(os.path.join(root, file))
        self.low_haze_rain_ids = derain_ids
        self.lhr_counter = 0
        self.num_lhr = len(self.low_haze_rain_ids)


    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def _get_gt_name(self, rainy_name):
        gt_name = 'data/CDD-11_train_100/clear/' + rainy_name.split('/')[-1]
        return gt_name

    def _get_random_prompt(self, de_id):
        key = self.prompts_key[de_id]
        return random.choice(self.prompts[key])

    def __getitem__(self, _):
        text_prompt = self._get_random_prompt(self.de_temp)

        if self.de_temp == 0:
            degrad_img = crop_img(np.array(Image.open(self.hazy_ids[self.haze_counter]).convert('RGB')), base=16)
            clean_name = self._get_gt_name(self.hazy_ids[self.haze_counter])
            clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)

            self.haze_counter = (self.haze_counter + 1) % self.num_haze
            if self.haze_counter == 0:
                random.shuffle(self.hazy_ids)

        elif self.de_temp == 1:
            degrad_img = crop_img(np.array(Image.open(self.rain_ids[self.rain_counter]).convert('RGB')), base=16)
            clean_name = self._get_gt_name(self.rain_ids[self.rain_counter])
            clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)

            self.rain_counter = (self.rain_counter + 1) % self.num_rain
            if self.rain_counter == 0:
                random.shuffle(self.rain_ids)

        elif self.de_temp == 2:
            degrad_img = crop_img(np.array(Image.open(self.low_ids[self.low_counter]).convert('RGB')), base=16)
            clean_name = self._get_gt_name(self.low_ids[self.low_counter])
            clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)

            self.low_counter = (self.low_counter + 1) % self.num_low
            if self.low_counter == 0:
                random.shuffle(self.low_ids)

        elif self.de_temp == 3:
            degrad_img = crop_img(np.array(Image.open(self.hazy_rain_ids[self.hr_counter]).convert('RGB')), base=16)
            clean_name = self._get_gt_name(self.hazy_rain_ids[self.hr_counter])
            clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)

            self.hr_counter = (self.hr_counter + 1) % self.num_hr
            if self.hr_counter == 0:
                random.shuffle(self.hazy_rain_ids)

        elif self.de_temp == 4:
            degrad_img = crop_img(np.array(Image.open(self.low_rain_ids[self.lr_counter]).convert('RGB')), base=16)
            clean_name = self._get_gt_name(self.low_rain_ids[self.lr_counter])
            clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)

            self.lr_counter = (self.lr_counter + 1) % self.num_lr
            if self.lr_counter == 0:
                random.shuffle(self.low_rain_ids)

        elif self.de_temp == 5:
            degrad_img = crop_img(np.array(Image.open(self.low_haze_ids[self.lh_counter]).convert('RGB')), base=16)
            clean_name = self._get_gt_name(self.low_haze_ids[self.lh_counter])
            clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)

            self.lh_counter = (self.lh_counter + 1) % self.num_lh
            if self.lh_counter == 0:
                random.shuffle(self.low_haze_ids)

        elif self.de_temp == 6:
            degrad_img = crop_img(np.array(Image.open(self.low_haze_rain_ids[self.lhr_counter]).convert('RGB')), base=16)
            clean_name = self._get_gt_name(self.low_haze_rain_ids[self.lhr_counter])
            clean_img = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)

            self.lhr_counter = (self.lhr_counter + 1) % self.num_lhr
            if self.lhr_counter == 0:
                random.shuffle(self.low_haze_rain_ids)

        degrad_patch_1, clean_patch_1 = random_augmentation(*self._crop_patch(degrad_img, clean_img))
        degrad_patch_2, clean_patch_2 = random_augmentation(*self._crop_patch(degrad_img, clean_img))

        clean_patch_1, clean_patch_2 = self.toTensor(clean_patch_1), self.toTensor(clean_patch_2)
        degrad_patch_1, degrad_patch_2 = self.toTensor(degrad_patch_1), self.toTensor(degrad_patch_2)

        self.de_temp = (self.de_temp + 1) % 7
        if self.de_temp == 0:
            random.shuffle(self.de_type)

        return [clean_name, 0], degrad_patch_1, degrad_patch_2, clean_patch_1, clean_patch_2, text_prompt

    def __len__(self):
        return 400 * len(self.args.de_type)


class DerainDehazeDataset(Dataset):
    def __init__(self, args, task="derain"):
        super(DerainDehazeDataset, self).__init__()
        self.ids = []
        self.task_idx = 0
        self.args = args
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

        self.task_dict = {'derain': 0, 'dehaze': 1}
        self.toTensor = ToTensor()

        self.set_dataset(task)

    def _init_input_ids(self):
        self.ids = []

        for root, dirs, files in os.walk(self.args.derain_path):
            for file in files:
                if os.path.splitext(file)[1].lower() in self.image_extensions and 'clear' not in root:
                    self.ids.append(os.path.join(root, file))

        self.length = len(self.ids)

    def _get_gt_path(self, degraded_name):
        gt_name = degraded_name.split('/')[0] + '/' + degraded_name.split('/')[1] + '/clear/' + degraded_name.split('/')[-1]
        return gt_name

    def set_dataset(self, task):
        self.task_idx = self.task_dict[task]
        self._init_input_ids()

    def __getitem__(self, idx):
        degraded_path = self.ids[idx]
        clean_path = self._get_gt_path(degraded_path)
        degradation = degraded_path.split('/')[2]

        text_prompt = ""

        if degradation == "haze":
            text_prompt = "Do Dehaze"

        elif degradation == "rain":
            text_prompt = "Do Derain."

        elif degradation == "low":
            text_prompt = "Do Brighter."

        elif degradation == "haze_rain":
            text_prompt = "Do Derain and Dehaze."

        elif degradation == "low_rain":
            text_prompt = "Do Brighter and Derain."

        elif degradation == "low_haze":
            text_prompt = "Do Brighter and Dehaze."

        elif degradation == "low_haze_rain":
            text_prompt = "Do Brighter and Dehaze and Derain."
            
        degraded_img = crop_img(np.array(Image.open(degraded_path).convert('RGB')), base=16)
        clean_img = crop_img(np.array(Image.open(clean_path).convert('RGB')), base=16)

        clean_img, degraded_img = self.toTensor(clean_img), self.toTensor(degraded_img)
        degraded_name = degraded_path.split('/')[-1][:-4]

        return [degraded_name], degradation, degraded_img, clean_img, text_prompt 

    def __len__(self):
        return self.length


class TestSpecificDataset(Dataset):
    def __init__(self, args):
        super(TestSpecificDataset, self).__init__()
        self.args = args
        self.degraded_ids = []
        self._init_clean_ids(args.test_path)

        self.toTensor = ToTensor()

    def _init_clean_ids(self, root):
        degraded_ids = []
        # 파일 목록 중 이미지 파일만 필터링
        name_list = os.listdir(root)
        name_list = [file for file in name_list if os.path.splitext(file)[1].lower() in self.image_extensions]
        self.degraded_ids += [root + id_ for id_ in name_list]

        self.num_img = len(self.degraded_ids)

    def __getitem__(self, idx):
        degraded_img = crop_img(np.array(Image.open(self.degraded_ids[idx]).convert('RGB')), base=16)
        name = self.degraded_ids[idx].split('/')[-1][:-4]

        degraded_img = self.toTensor(degraded_img)

        return [name], degraded_img

    def __len__(self):
        return self.num_img
