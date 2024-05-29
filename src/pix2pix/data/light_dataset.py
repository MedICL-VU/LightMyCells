import os
from os import listdir
from os.path import join
import torch
import copy
import numpy as np
from data.base_dataset import BaseDataset
import random
from aicsimageio.readers.bioformats_reader import BioFile
import tifffile
from monai.transforms import *


class LightDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.image_dict = get_dict(opt.dataroot)
        self.m_list, self.n_list, self.t_list, self.a_list = self.get_class_dicts()
        self.BF_list, self.PC_list, self.DIC_list = self.get_input_dicts()
    
        # filter to single input modality ***
        data_list = []
        if opt.input_modality == 'BF':
            data_list = self.BF_list
        if opt.input_modality == 'PC':
            data_list = self.PC_list
        if opt.input_modality == 'DIC':
            data_list = self.DIC_list

        self.m_list = [x for x in self.m_list if x in data_list]
        self.n_list = [x for x in self.n_list if x in data_list]
        self.t_list = [x for x in self.t_list if x in data_list]
        self.a_list = [x for x in self.a_list if x in data_list]
        
        self.transform = LightTransform()
    
    def __len__(self):
        return len(self.image_dict)

    def __getitem__(self, index):
        # equal number of sampling
        if index % 4 == 0:
            sample = random.choice(self.m_list)
        elif index % 4 == 1:
            sample = random.choice(self.n_list)
        elif index % 4 == 2:
            sample = random.choice(self.t_list)
        elif index % 4 == 3:
            if len(self.a_list) == 0:
                sample = random.choice(self.t_list)
            else:
                sample = random.choice(self.a_list)
        data = copy.deepcopy(sample)
        data_dict = data['output']
        input_key = random.choice(list(data['input'].keys()))
        data_dict['input'] = data['input'][input_key]
        data_dict = self.transform(data_dict)
        return data_dict

    def get_class_dicts(self):
        m_list, n_list, t_list, a_list = [], [], [], []
        image_keys = self.image_dict.keys()
        for key in image_keys:
            keys = self.image_dict[key]['output'].keys()
            if 'Mitochondria' in keys:
                m_list.append(self.image_dict[key])
            if 'Nucleus' in keys:
                n_list.append(self.image_dict[key])
            if 'Tubulin' in keys:
                t_list.append(self.image_dict[key])
            if 'Actin' in keys:
                a_list.append(self.image_dict[key])
        return m_list, n_list, t_list, a_list

    def get_input_dicts(self):
        BF_list, PC_list, DIC_list = [], [], []
        image_keys = self.image_dict.keys()
        for key in image_keys:
            # breakpoint()
            values = self.image_dict[key]['input'].values()
            if 'BF' in str(values):
                BF_list.append(self.image_dict[key])
            if 'PC' in str(values):
                PC_list.append(self.image_dict[key])
            if 'DIC' in str(values):
                DIC_list.append(self.image_dict[key])
        return BF_list, PC_list, DIC_list

    # def get_class_dicts(self):
    #     m_list, n_list, t_list, a_list = [], [], [], []
    #     for i in range(len(self.image_dict)):
    #         keys = self.image_dict[i]['output'].keys()
    #         if 'Mitochondria' in keys:
    #             m_list.append(self.image_dict[i])
    #         if 'Nucleus' in keys:
    #             n_list.append(self.image_dict[i])
    #         if 'Tubulin' in keys:
    #             t_list.append(self.image_dict[i])
    #         if 'Actin' in keys:
    #             a_list.append(self.image_dict[i])
    #     return m_list, n_list, t_list, a_list

    # def get_input_dicts(self):
    #     BF_list, PC_list, DIC_list = [], [], []
    #     for i in range(len(self.image_dict)):
    #         values = self.image_dict[i]['input'].values()
    #         if 'BF' in str(values):
    #             BF_list.append(self.image_dict[i])
    #         if 'PC' in str(values):
    #             PC_list.append(self.image_dict[i])
    #         if 'DIC' in str(values):
    #             DIC_list.append(self.image_dict[i])
    #     return BF_list, PC_list, DIC_list


def LightTransform():
    name_list = ['input', 'Mitochondria', 'Nucleus', 'Tubulin', 'Actin']
    mask_list = ['Mitochondria_mask', 'Nucleus_mask', 'Tubulin_mask', 'Actin_mask']
    return Compose([        
        # GetInputCoded(keys='input'),
        GetOutputCoded(keys=name_list[1:]),
        LoadBioImaged(keys=name_list),
        AddChanneld(keys=name_list, allow_missing_keys=True),

        # data preprocessing & normalization
        GetWeightingMaskd(keys=name_list[1:]), 
        ScaleIntensityRanged(
            keys=name_list, a_min=0, a_max=np.iinfo(np.uint16).max, 
            b_min=-1, b_max=1, clip=False, allow_missing_keys=True),

        # NormalizeIntensityd(keys=['input']),
        Placeholderd(keys=name_list+mask_list),

        # intensity augmentation
        # RandAdjustContrastd(keys=['input'], prob=0.1, gamma=(0.8, 1.2), allow_missing_keys=True),
        
        # spatial augmentation
        RandSpatialCropd(
            keys=name_list+mask_list, roi_size=(512, 512), random_center=True, 
            random_size=False, allow_missing_keys=True),
        RandRotate90d(keys=name_list+mask_list, max_k=3, prob=0.5),
        RandFlipd(keys=name_list+mask_list, prob=0.5, spatial_axis=0, allow_missing_keys=True),
        RandFlipd(keys=name_list+mask_list, prob=0.5, spatial_axis=1, allow_missing_keys=True),
        CastToTyped(keys=name_list+mask_list, dtype=np.float32, allow_missing_keys=True),
        ToTensord(keys=name_list+mask_list, allow_missing_keys=True)])


class LoadBioImaged(MapTransform):
    def __init__(self, keys) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if key in data.keys():
                data[key] = BioFile(data[key]).to_numpy()[0, 0, 0]
        return data


# class LoadBioImaged(MapTransform):
#     def __init__(self, keys) -> None:
#         MapTransform.__init__(self, keys)
#         self.keys = keys

#     def __call__(self, data):
#         for key in self.keys:
#             if key in data.keys():
#                 with tifffile.TiffFile(data[key]) as tif:
#                     data[key] = tif.asarray().astype('float32')
#         return data
        

class GetWeightingMaskd(MapTransform):
    def __init__(self, keys) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if key in data.keys():
                data[f'{key}_mask'] = get_weighting_mask(data[key], pmin=2, pmax=99.8)
        return data


class NormalizeBioImaged(MapTransform):
    def __init__(self, keys) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if key in data.keys():
                data[key] = percentile_normalization(data[key], pmin=2, pmax=99.8)
        return data


class Placeholderd(MapTransform):
    def __init__(self, keys) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if key not in data.keys():
                data[key] = np.ones_like(data['input'])
        return data


class GetInputCoded(MapTransform):
    def __init__(self, keys) -> None:
        MapTransform.__init__(self, keys)

    def __call__(self, data):
        if '_BF_' in data['input']:
            data['input_code'] = torch.tensor([0,0,1])
        elif '_DIC_' in data['input']:
            data['input_code'] = torch.tensor([0,1,0])
        elif '_PC_' in data['input']:
            data['input_code'] = torch.tensor([1,0,0])
        return data


class GetOutputCoded(MapTransform):
    def __init__(self, keys) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys

    def __call__(self, data):
        vec = torch.tensor([0, 0, 0, 0])
        for i, key in enumerate(self.keys):
            if key in data.keys():
                vec[i] = 1
        data['output_code'] = vec
        # print(data.keys(), data['output_code'])
        return data


def percentile_normalization(image, pmin=2, pmax=99.8, axis=None, dtype=np.uint16):
    if not (np.isscalar(pmin) and np.isscalar(pmax) and 0 <= pmin < pmax <= 100):
        raise ValueError("Invalid values for pmin and pmax")
    low_p  = np.percentile(image, pmin, axis=axis, keepdims=True)
    high_p = np.percentile(image, pmax, axis=axis, keepdims=True)

    if low_p == high_p:
        img_norm = image
        # print(f"Same min {low_p} and high {high_p}, image may be empty")
    else:
        dtype_max = np.iinfo(dtype).max
        img_norm = dtype_max * (image - low_p) / ( high_p - low_p )
        img_norm = img_norm.astype(dtype)
    return img_norm


def get_weighting_mask(image, pmin=2, pmax=99.8, axis=None):
    low_p  = np.percentile(image, pmin, axis=axis, keepdims=True)
    high_p = np.percentile(image, pmax, axis=axis, keepdims=True)
    mask = np.ones(image.shape)
    mask[(image <= low_p) | (image >= high_p)] = 0.1
    return mask


#===================== Helper functions =======================


def get_image_number(filename):
    #Return image number of a given image filename
    filename=filename[6:]
    return int(filename[0:filename.find("_")])


def is_input_image(filename):
    #Return True if image filename is BF or DIC or PC  
    if filename.find("_BF")>=0 or filename.find("_DIC")>=0 or filename.find("_PC")>=0:
        return True
    return False


def get_channel_type(filename):
    #Return the channel type of a given image filename
    filename=filename[6:]
    filename=filename[filename.find("_")+1:]
    if filename.find("_")>=0: 
        return filename[0:filename.find("_")]
    return filename[0:filename.find(".")]


def get_z(filename):
    #Return the z value  of a given image filename
    if filename.find("_z")>=0: 
        return int(filename[filename.find("_z")+2:filename.find(".")])
    print(" Error miss Z information in "+filename)
    quit()


# Create Image files dictionary  
# def get_dict(db_path):
#     images={}
#     for study in listdir(db_path):
#         if study not in ['Study_8', 'Study_20', 'Study_28']:
#         #     continue
#         # print(" add images from "+study)
#             for filename in listdir(join(db_path,study)):
#                 number=get_image_number(filename)
#                 if number  not in images :  images[number]={}

#                 if is_input_image(filename):
#                     if 'input' not in images[number]: images[number]['input']={}
#                     images[number]['input'][get_z(filename)]=join(db_path,study,filename)
#                 else: 
#                     if 'output' not in images[number]: images[number]['output']={}
#                     images[number]['output'][get_channel_type(filename)]=join(db_path, study, filename)
#     print(f'Finish loading dictionary of {len(images)} images...')
#     return images


def get_dict(db_path):
    images={}
    for study in listdir(db_path):
        # print(" add images from "+study)
        for filename in listdir(join(db_path,study)):
            number=get_image_number(filename)
            if number  not in images :  images[number]={}

            if is_input_image(filename):
                if 'input' not in images[number]: images[number]['input']={}
                images[number]['input'][get_z(filename)]=join(db_path,study,filename)
            else: 
                if 'output' not in images[number]: images[number]['output']={}
                images[number]['output'][get_channel_type(filename)]=join(db_path, study, filename)
    print(f'Finish loading dictionary of {len(images)} images...')
    return images