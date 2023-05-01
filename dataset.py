#  Copyright (c) 2021 Mandar Gogate, All rights reserved.

import logging
import math
import os, pdb
import random
from os.path import join

import librosa
import numpy as np
import torch
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset

from config import GRID_IMAGES_ROOT_sq, GRID_ROOT, NOISE_ROOT, SEED, img_height, img_width, nb_channels, sampling_rate, stft_size, window_shift, window_size
from utils.data import get_images
from utils.generic import subsample_list


def get_transform():
    transform_list = [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


test_transform = get_transform()


class GridDataset(Dataset):
    def __init__(self, speakers, noise_types, raw_data_root, noise_root, images_root, shuffle=True, seed=SEED, audio_prefix="audio_16000", subsample=1, add_channel_dim=False, a_only=True, mode='SS', return_stft=False):
        self.return_stft = return_stft
        self.a_only = a_only
        self.mode = mode
        self.images_root = images_root
        self.add_channel_dim = add_channel_dim
        self.speakers = speakers
        self.noise_types = noise_types
        self.raw_data_root = raw_data_root
        self.noise_root = noise_root
        self.audio_prefix = audio_prefix
        self.files_list = self.build_files_list
        self.noise_list = self.build_noise_list
        self.rgb = True if nb_channels == 3 else False
        if shuffle:
            random.seed(SEED)
            random.shuffle(self.files_list)
            random.shuffle(self.noise_list)
        if subsample != 1:
            self.files_list = subsample_list(self.files_list, sample_rate=subsample)
            self.noise_list = subsample_list(self.noise_list, sample_rate=subsample)
        logging.info("Found {} utterances".format(len(self.files_list)))
        logging.info("Found {} utterances".format(len(self.noise_list))) 
        self.data_count = len(self.files_list)
        self.batch_index = 0
        self.total_batches_seen = 0
        self.batch_input = {"noisy": None}
        self.index = 0
        self.max_len = len(self.files_list)
        self.max_cache = 0
        self.seed = seed
        self.window = "hann"
        self.fading = False

    @property
    def build_files_list(self):
        files_list = []
        for speaker in self.speakers:
            clean_root = join(self.raw_data_root,self.audio_prefix, speaker)
            # pdb.set_trace()
            for audio_file in os.listdir(clean_root):
                clean_file = join(clean_root, audio_file)
                file_id = audio_file.split(".")[0]
                files_list.append([speaker, file_id, clean_file])
        return files_list
     
    @property 
    def build_noise_list(self):
        noise_list = []
        for noise in self.noise_types:
            for audio_file in os.listdir(self.noise_root):
                #pdb.set_trace()
                noise_file = join(self.noise_root, audio_file)
                noise_name = audio_file.split(".")[0]
                noise_list.append([noise_name, noise_file])
        return noise_list

    def __len__(self):
        if self.return_stft:
            return len(self.files_list)
        else:
            return len(self.files_list) * 2

    def __getitem__(self, idx):
        data = {}
        if self.mode == 'SS':
            (speaker, file_id, clean_file), (_, _, noise_file) = random.sample(self.files_list, 2)
        elif self.mode == 'SE':
            (speaker, file_id, clean_file), (_, _, _) = random.sample(self.files_list, 2)
            (_,noise_file), (_, _) = random.sample(self.noise_list, 2) 
        # pdb.set_trace()
        if not self.a_only:
            images_root = join(self.images_root, speaker, file_id)
            data["lip_images"] = self.get_lip_images(images_root)
        if self.return_stft:
            data["noisy_audio_spec"], data["mask"], data["clean"], data["noisy_stft"] = self.get_audiofeat(clean_file, noise_file)
        else:
            data["noisy_audio_spec"], data["mask"] = self.get_audiofeat(clean_file, noise_file)
        return data

    def get_noisy_features(self, noisy):
        audio_stft = librosa.stft(noisy, win_length=window_size, n_fft=stft_size, hop_length=window_shift, window=self.window, center=True)
        if self.add_channel_dim:
            return np.abs(audio_stft).astype(np.float32)[np.newaxis, ...]
        else:
            return np.abs(audio_stft).astype(np.float32)

    def get_lip_images(self, images_root, rgb=False):
        lip_image = np.zeros((64, img_height, img_width)).astype(np.float32)
        # pdb.set_trace()
        try:
            img = get_images(images_root+"/face.pt", rgb=rgb)
            # img = None
            # pdb.set_trace()
            if img is not None:
                img = img.astype(np.float32)
                img = img / 255
                mean = [0.5]
                std = [0.5]
                img = (img - mean) / std
                if lip_image.shape[0] <= img.shape[0]:
                    lip_image = img[:lip_image.shape[0]]
                else:
                    lip_image[:img.shape[0]] = img
        except Exception as e:
            print(e)
            pass
        # pdb.set_trace()
        return lip_image[np.newaxis, ...].astype(np.float32)

    def get_audiofeat(self, clean_file, noise_file):
        noise, _ = librosa.load(noise_file, sr=sampling_rate)
        clean, _ = librosa.load(clean_file, sr=sampling_rate)
        if noise.shape[-1]<= clean.shape[-1]:
            ratio = int(clean.shape[-1]//noise.shape[-1])
            noise = noise.repeat(ratio+1)
            # print(noise)
        clean, noise = clean[:40900], noise[:40900]
        if noise.shape[0] > clean.shape[0]:
            clean = np.pad(clean, pad_width=[0, noise.shape[0] - clean.shape[0]], mode="constant")
        else:
            noise = np.pad(noise, pad_width=[0, clean.shape[0] - noise.shape[0]], mode="constant")
        noise_db = random.randint(0, 20)
        clean_power = np.linalg.norm(clean, 2)
        noise_power = np.linalg.norm(noise, 2)
        snr = math.exp(noise_db / 10)
        scale = snr * noise_power / clean_power
        noisy = (scale * clean + noise) / 2
        if self.return_stft:
            clean_audio = clean
            noisy_stft = librosa.stft(noisy, win_length=window_size, n_fft=stft_size, hop_length=window_shift, window=self.window, center=True)
            return self.get_noisy_features(noisy), self.get_noisy_features(
                clean), clean_audio[:-100], noisy_stft

        else:
            return self.get_noisy_features(noisy), self.get_noisy_features(clean)

class GridDataset_test(Dataset):
    def __init__(self, snrs, speakers, noise_types, raw_data_root, noise_root, images_root, shuffle=True, seed=SEED, audio_prefix="audio_16000", subsample=1, add_channel_dim=False, a_only=True, mode='SS', return_stft=False):
        self.return_stft = return_stft
        self.a_only = a_only
        self.mode = mode
        self.snrs = snrs
        self.images_root = images_root
        self.add_channel_dim = add_channel_dim
        self.speakers = speakers
        self.noise_types = noise_types
        self.raw_data_root = raw_data_root
        self.noise_root = noise_root
        self.audio_prefix = audio_prefix
        self.files_list = self.build_files_list
        self.noise_list = self.build_noise_list
        self.rgb = True if nb_channels == 3 else False
        if shuffle:
            random.seed(SEED)
            random.shuffle(self.files_list)
            random.shuffle(self.noise_list)
        if subsample != 1:
            self.files_list = subsample_list(self.files_list, sample_rate=subsample)
            self.noise_list = subsample_list(self.noise_list, sample_rate=subsample)
        logging.info("Found {} utterances".format(len(self.files_list)))
        logging.info("Found {} utterances".format(len(self.noise_list))) 
        self.data_count = len(self.files_list)
        self.batch_index = 0
        self.total_batches_seen = 0
        self.batch_input = {"noisy": None}
        self.index = 0
        self.max_len = len(self.files_list)
        self.max_cache = 0
        self.seed = seed
        self.window = "hann"
        self.fading = False

    @property
    def build_files_list(self):
        files_list = []
        for speaker in self.speakers:
            clean_root = join(self.raw_data_root,self.audio_prefix, speaker)
            # pdb.set_trace()
            for audio_file in os.listdir(clean_root):
                clean_file = join(clean_root, audio_file)
                file_id = audio_file.split(".")[0]
                files_list.append([speaker, file_id, clean_file])
        return files_list
     
    @property 
    def build_noise_list(self):
        noise_list = []
        for noise in self.noise_types:
            for audio_file in os.listdir(self.noise_root):
                #pdb.set_trace()
                noise_file = join(self.noise_root, audio_file)
                noise_name = audio_file.split(".")[0]
                noise_list.append([noise_name, noise_file])
        return noise_list

    def __len__(self):
        if self.return_stft:
            return len(self.files_list)
        else:
            return len(self.files_list) * 2

    def __getitem__(self, idx):
        data = {}
        if self.mode == 'SS':
            (speaker, file_id, clean_file), (_, _, noise_file) = random.sample(self.files_list, 2)
        elif self.mode == 'SE':
            (speaker, file_id, clean_file), (_, _, _) = random.sample(self.files_list, 2)
            (_,noise_file), (_, _) = random.sample(self.noise_list, 2) 
        # pdb.set_trace()
        if not self.a_only:
            images_root = join(self.images_root, speaker, file_id)
            data["lip_images"] = self.get_lip_images(images_root)
        if self.return_stft:
            data["noisy_audio_spec"], data["mask"], data["clean"], data["noisy_stft"] = self.get_audiofeat(clean_file, noise_file)
        else:
            data["noisy_audio_spec"], data["mask"] = self.get_audiofeat(clean_file, noise_file)
        return data

    def get_noisy_features(self, noisy):
        audio_stft = librosa.stft(noisy, win_length=window_size, n_fft=stft_size, hop_length=window_shift, window=self.window, center=True)
        if self.add_channel_dim:
            return np.abs(audio_stft).astype(np.float32)[np.newaxis, ...]
        else:
            return np.abs(audio_stft).astype(np.float32)

    def get_lip_images(self, images_root, rgb=False):
        lip_image = np.zeros((64, img_height, img_width)).astype(np.float32)
        # pdb.set_trace()
        try:
            img = get_images(images_root+"/face.pt", rgb=rgb)
            # img = None
            # pdb.set_trace()
            if img is not None:
                img = img.astype(np.float32)
                img = img / 255
                mean = [0.5]
                std = [0.5]
                img = (img - mean) / std
                if lip_image.shape[0] <= img.shape[0]:
                    lip_image = img[:lip_image.shape[0]]
                else:
                    lip_image[:img.shape[0]] = img
        except Exception as e:
            print(e)
            pass
        # pdb.set_trace()
        return lip_image[np.newaxis, ...].astype(np.float32)

    def get_audiofeat(self, clean_file, noise_file):
        noise, _ = librosa.load(noise_file, sr=sampling_rate)
        clean, _ = librosa.load(clean_file, sr=sampling_rate)
        # if noise.shape[-1]<= clean.shape[-1]:
        #     ratio = int(clean.shape[-1]//noise.shape[-1])
        #     noise = noise.repeat(ratio+1)
        # print(noise.shape)
        # pdb.set_trace()
        clean, noise = clean[:40900], noise[:40900]
        if noise.shape[0] > clean.shape[0]:
            clean = np.pad(clean, pad_width=[0, noise.shape[0] - clean.shape[0]], mode="constant")
        else:
            noise = np.pad(noise, pad_width=[0, clean.shape[0] - noise.shape[0]], mode="constant")
        noise_db = self.snrs
        clean_power = np.linalg.norm(clean, 2)
        noise_power = np.linalg.norm(noise, 2)
        snr = math.exp(noise_db / 10)
        scale = snr * noise_power / clean_power
        noisy = (scale * clean + noise) / 2
        noisy = noisy/max(abs(noisy))
        # import soundfile as sf
        # sf.write('testt.wav', noisy, samplerate=sampling_rate)
        if self.return_stft:
            clean_audio = clean
            noisy_stft = librosa.stft(noisy, win_length=window_size, n_fft=stft_size, hop_length=window_shift, window=self.window, center=True)
            return self.get_noisy_features(noisy), self.get_noisy_features(
                clean), clean_audio[:-100], noisy_stft

        else:
            return self.get_noisy_features(noisy), self.get_noisy_features(clean)

class GridDataModule(LightningDataModule):
    def __init__(self, snrs, batch_size=4, add_channel_dim=False, mode = "SS", a_only=False):
        super(GridDataModule, self).__init__()
        train_speakers_ids, val_speakers_ids, test_speakers_ids = [4, 7, 11, 16, 23, 24, 25, 29, 31, 33, 34, 3, 5, 6, 9, 10, 13, 14, 17, 19, 26, 27, 28], [1, 32, 2, 30], [18, 20, 22, 26]
        train_speakers = ["S{}".format(speaker) for speaker in train_speakers_ids]
        val_speakers = ["S{}".format(speaker) for speaker in val_speakers_ids]
        test_speakers = ["S{}".format(speaker) for speaker in test_speakers_ids]
        train_noise_names, val_noise_names, test_noise_names = [70, 38, 26, 90, 44, 83, 18, 88, 11, 95, 64, 51, 91, 9, 53, 33, 40, 74, 25, 8, 56, 6, 71, 27, 87, 29, 0, 43, 34, 60, 55, 10, 19, 77, 81, 54, 89, 20, 12, 17, 30, 82, 94, 5, 42, 75, 23, 21, 92, 4, 59, 86, 47, 78, 85, 35, 65, 80, 31, 84, 61, 69, 49, 98, 28, 50, 16, 36, 37, 93, 24, 7, 63, 79, 32, 1, 41, 39, 72, 76], [99, 46, 3, 96, 15, 68, 13, 73, 62, 66], [58, 45, 67, 48, 97, 14, 57, 2, 22, 52]
        train_noise= ["n{}".format(noise) for noise in train_noise_names]
        val_noise = ["n{}".format(noise) for noise in val_noise_names]
        test_noise= ["n{}".format(noise) for noise in test_noise_names] 
        self.train_dataset = GridDataset(train_speakers, train_noise, GRID_ROOT, NOISE_ROOT, GRID_IMAGES_ROOT_sq, add_channel_dim=add_channel_dim, mode=mode, a_only=a_only)
        self.val_dataset = GridDataset(val_speakers, val_noise, GRID_ROOT, NOISE_ROOT, GRID_IMAGES_ROOT_sq, add_channel_dim=add_channel_dim, mode=mode, a_only=a_only, return_stft=True)
        self.test_dataset = GridDataset_test(snrs, test_speakers, test_noise, GRID_ROOT, NOISE_ROOT, GRID_IMAGES_ROOT_sq, add_channel_dim=add_channel_dim, mode=mode, a_only=a_only, return_stft=True)
        self.batch_size = batch_size

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)


if __name__ == '__main__':
    mask = "IRM"
    train_speakers_ids, val_speakers_ids, test_speakers_ids = [4, 7, 11, 16, 23, 24, 25, 29, 31, 33, 34, 3, 5, 6, 9, 10, 13, 14, 17, 19, 26, 27, 28], [1, 32, 2, 30], [18, 20, 22, 26]
    train_speakers = ["S{}".format(speaker) for speaker in train_speakers_ids]
    val_speakers = ["S{}".format(speaker) for speaker in val_speakers_ids]
    test_speakers = ["S{}".format(speaker) for speaker in test_speakers_ids]
    train_noise_names, val_noise_names, test_noise_names = [70, 38, 26, 90, 44, 83, 18, 88, 11, 95, 64, 51, 91, 9, 53, 33, 40, 74, 25, 8, 56, 6, 71, 27, 87, 29, 0, 43, 34, 60, 55, 10, 19, 77, 81, 54, 89, 20, 12, 17, 30, 82, 94, 5, 42, 75, 23, 21, 92, 4, 59, 86, 47, 78, 85, 35, 65, 80, 31, 84, 61, 69, 49, 98, 28, 50, 16, 36, 37, 93, 24, 7, 63, 79, 32, 1, 41, 39, 72, 76], [58, 45, 67, 48, 97, 14, 57, 2, 22, 52], [99, 46, 3, 96, 15, 68, 13, 73, 62, 66]
    train_noise= ["n{}".format(noise) for noise in train_noise_names]
    val_noise = ["n{}".format(noise) for noise in val_noise_names]
    test_noise= ["n{}".format(noise) for noise in test_noise_names] 
    dataset = GridDataset_test(0.0, test_speakers, test_noise, raw_data_root=GRID_ROOT, noise_root=NOISE_ROOT, images_root=GRID_IMAGES_ROOT_sq, mode = "SS", a_only=False, return_stft=True)
    for i in range(10):
        data = dataset[i]

        for k,v in data.items():
            pdb.set_trace()
            # print(k, v.shape, np.min(v), np.max(v), np.mean(v))
    # print(dataset.files_list[:10])
