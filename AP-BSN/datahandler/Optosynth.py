import os

import h5py

from src.datahandler.denoise_dataset import DenoiseDataSet
from . import regist_dataset


@regist_dataset
class Optosynth(DenoiseDataSet):
    '''
    dataset class for prepared Optosynth dataset which is cropped with overlap.
    [using size 100x100 with 16 overlapping]
    (The path needs to be changed when preparing the data)
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _scan(self):
        self.dataset_path = os.path.join(self.dataset_dir, 'optosynth/train')
        assert os.path.exists(self.dataset_path), 'There is no dataset %s'%self.dataset_path
        for root, _, files in os.walk(os.path.join(self.dataset_path, 'RN')):
            self.img_paths = files

    def _load_data(self, data_idx):
        file_name = self.img_paths[data_idx]

        noisy_img = self._load_img(os.path.join(self.dataset_path, 'RN', file_name))

        return {'real_noisy': noisy_img}


@regist_dataset
class Optosynth_val(DenoiseDataSet):
    '''
    dataset class for testing for prepared Optosynth dataset which is cropped with overlap.
    [using size 100x100 with 16 overlapping]
    (The path needs to be changed when preparing the data)
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _scan(self):
        self.dataset_path = os.path.join(self.dataset_dir, 'optosynth/test')
        assert os.path.exists(self.dataset_path), 'There is no dataset %s'%self.dataset_path
        for root, _, files in os.walk(os.path.join(self.dataset_path, 'RN')):
            self.img_paths = files

    def _load_data(self, data_idx):
        file_name_RN = self.img_paths[data_idx]
        file_name_CL = file_name_RN.replace('raw__', '')

        noisy_img = self._load_img(os.path.join(self.dataset_path, 'RN', file_name_RN))
        clean = self._load_img(os.path.join(self.dataset_path, 'CL', file_name_CL))

        return {'clean': clean, 'real_noisy': noisy_img}

