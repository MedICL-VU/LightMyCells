import os
from os import listdir
from os.path import join
import torch
import copy
import numpy as np
from data.base_dataset import BaseDataset
import random
from aicsimageio.readers.bioformats_reader import BioFile
from monai.transforms import *


class ClassifierDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.image_dict = get_dict(opt.dataroot)
        self.BF_list, self.PC_list, self.DIC_list = self.get_input_dicts()        
        self.transform = ClassifyTransform()
    
    def __len__(self):
        return int(2000)  # len(self.image_dict)

    def __getitem__(self, index):
        # Equal number of sampling
        if index % 3 == 0:
            sample = random.choice(self.BF_list)
        elif index % 3 == 1:
            sample = random.choice(self.PC_list)
        elif index % 3 == 2:
            sample = random.choice(self.DIC_list)

        data_dict = {}
        input_key = random.choice(list(sample['input'].keys()))
        data_dict['input'] = sample['input'][input_key]
        data_dict = self.transform(data_dict)
        return data_dict

    def get_input_dicts(self):
        BF_list, PC_list, DIC_list = [], [], []
        for i in range(len(self.image_dict)):
            values = self.image_dict[i]['input'].values()
            if 'BF' in str(values):
                BF_list.append(self.image_dict[i])
            if 'PC' in str(values):
                PC_list.append(self.image_dict[i])
            if 'DIC' in str(values):
                DIC_list.append(self.image_dict[i])
        return BF_list, PC_list, DIC_list


def ClassifyTransform():
    return Compose([        
        GetLabeld(keys='input'),
        LoadBioImaged(keys=['input']),
        AddChanneld(keys=['input', 'label']),

        # data preprocessing & normalization
        ScaleIntensityRanged(
            keys='input', a_min=0, a_max=np.iinfo(np.uint16).max, 
            b_min=0, b_max=1, clip=False, allow_missing_keys=True),

        # intensity augmentation
        RandGaussianSmoothd(keys=['input'], prob=0.1),
        RandScaleIntensityd(keys=['input'], prob=0.1, factors=0.2),

        # spatial augmentation
        RandAffined(keys='input', spatial_size=None, 
            prob=0.2, 
            rotate_range=(np.pi/4, np.pi/4), 
            translate_range=(25, 25),
            scale_range=(0.1, 0.1),
            padding_mode='zeros'),
        
        Resized(keys='input', spatial_size=(320, 320)),
        RandFlipd(keys='input', prob=0.5, spatial_axis=0, allow_missing_keys=True),
        RandFlipd(keys='input', prob=0.5, spatial_axis=1, allow_missing_keys=True),
        CastToTyped(keys='input', dtype=np.float32, allow_missing_keys=True),
        ToTensord(keys='input', allow_missing_keys=True)])


class LoadBioImaged(MapTransform):
    def __init__(self, keys) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if key in data.keys():
                data[key] = BioFile(data[key]).to_numpy()[0, 0, 0].astype('float32')
        return data


class GetLabeld(MapTransform):
    def __init__(self, keys) -> None:
        MapTransform.__init__(self, keys)

    def __call__(self, data):
        if '_BF_' in data['input']:
            data['label'] = torch.tensor(0)
        elif '_PC_' in data['input']:
            data['label'] = torch.tensor(1)
        elif '_DIC_' in data['input']:
            data['label'] = torch.tensor(2)
        return data


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