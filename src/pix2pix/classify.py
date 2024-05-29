import os
from os import listdir
from os.path import join
import os.path as osp
import torch
import numpy as np
import random
from glob import glob
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from monai.transforms import *
import util.util as util
from tqdm import tqdm 
from aicsimageio.readers.bioformats_reader import BioFile


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


def load_model(opt):
    opt.num_threads = 0  
    opt.batch_size = 1  
    opt.serial_batches = True
    model = create_model(opt)
    model.setup(opt)
    if opt.eval:
        model.eval()
    return model


def ClassifyTransform():
    return Compose([        
        LoadBioImaged(keys=['input']),
        AddChanneld(keys=['input']),
        # data preprocessing & normalization
        ScaleIntensityRanged(
            keys='input', a_min=0, a_max=np.iinfo(np.uint16).max, 
            b_min=0, b_max=1, clip=False),
        Resized(keys='input', spatial_size=(256, 256)),
        CastToTyped(keys='input', dtype=np.float32),
        ToTensord(keys='input')])


class LoadBioImaged(MapTransform):
    def __init__(self, keys) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if key in data.keys():
                data[key] = BioFile(data[key]).to_numpy()[0, 0, 0].astype('float32')
        return data


def get_input_dicts():
    BF_list, PC_list, DIC_list = [], [], []
    for i in range(len(image_dict)):
        values = image_dict[i]['input'].values()
        if 'BF' in str(values):
            BF_list.append(image_dict[i])
        if 'PC' in str(values):
            PC_list.append(image_dict[i])
        if 'DIC' in str(values):
            DIC_list.append(image_dict[i])
    return BF_list, PC_list, DIC_list


def classify(image_path: str):  
    """BF: 0, PC: 1, DIC: 2"""
    opt = TestOptions().parse()
    opt.model = 'classifier'
    model = load_model(opt).netG
    data_dict = {'input': image_path}
    data_dict = ClassifyTransform()(data_dict)
    with torch.no_grad():
        pred = torch.argmax(model(data_dict), 1).squeeze().cpu().numpy()
    return pred


# validation
if __name__ == '__main__':
    opt = TestOptions().parse()
    model = load_model(opt).netG
    image_dict = get_dict(opt.dataroot)
    BF_list, PC_list, DIC_list = get_input_dicts()        

    save_dir = osp.join(opt.checkpoints_dir, opt.name, 'result')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    with torch.no_grad():
        for sample in BF_list:
            data_dict = {}
            input_key = random.choice(list(sample['input'].keys()))
            data_dict['input'] = sample['input'][input_key]
            data_dict = ClassifyTransform()(data_dict)
            pred = model(data_dict)
    
                
