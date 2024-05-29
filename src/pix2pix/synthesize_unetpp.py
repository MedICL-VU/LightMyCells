import os
from os import listdir
from time import time
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
import tifffile
import xmltodict
from monai.inferers import sliding_window_inference


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


def LightTransform():
    return Compose([        
        LoadBioImaged(keys=['input']),
        AddChanneld(keys=['input']),
        ScaleIntensityRanged(
            keys=['input'], a_min=0, a_max=np.iinfo(np.uint16).max, 
            b_min=-1, b_max=1, clip=False, allow_missing_keys=True),
        NormalizeIntensityd(keys=['input']),  # z-score normalization
        CastToTyped(keys=['input'], dtype=np.float32),
        ToTensord(keys=['input'])])


class LoadBioImaged(MapTransform):
    def __init__(self, keys) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if key in data.keys():
                with tifffile.TiffFile(data[key]) as tif:
                    data[key] = tif.asarray().astype('float32')
        return data


def save_image(location, array, metadata):
    #Save each predicted images with the required metadata
    print(" --> save "+str(location))
    pixels = xmltodict.parse(metadata)["OME"]["Image"]["Pixels"]
    physical_size_x = float(pixels["@PhysicalSizeX"])
    physical_size_y = float(pixels["@PhysicalSizeY"])
    
    tifffile.imwrite(location,
                     array,
                     description=metadata,
                     resolution=(physical_size_x, physical_size_y),
                     metadata=pixels,
                     tile=(128, 128),
                     )


def synthesize(image_path: str, save_dir: str, save=True):  
    """BF: 0, PC: 1, DIC: 2"""
    opt = TestOptions().parse()
    opt.netG = 'res_multi_decoder'
    opt.norm = 'batch'

    with tifffile.TiffFile(image_path) as tif:
        metadata = tif.ome_metadata
    description = xmltodict.parse(metadata)
    tl= description["OME"]['Image']["Pixels"]["Channel"]["@Name"]

    if tl == 'BF':
        opt.name = 'BF_multi_decoder'
    elif tl == 'PC':
        opt.name = 'PC_multi_decoder'
    elif tl == 'DIC':
        opt.name = 'DIC_multi_decoder'
    
    model = load_model(opt).netG
    data_dict = LightTransform()({'input': image_path})

    # Test-time augmentation
    output_A, output_B, output_C, output_D = [], [], [], [] 

    with torch.no_grad():
        for k in range(4): # customize from 1->4
            input_data = data_dict['input'].squeeze(0)
            # test-time augmentation: rotate
            input_data = torch.rot90(input_data, k=k).unsqueeze(0)
            # inference
            A, B, C, D = sliding_window_inference(
                inputs=input_data.unsqueeze(0).cuda(), 
                roi_size=(512, 512), 
                sw_batch_size=24, 
                predictor=model,
                overlap=0.8,
                mode='gaussian',
                sigma_scale=0.125,
                padding_mode='constant',
                cval=-1,
                is_train=False)[0]

            # test-time augmentation: rotate back
            A = torch.rot90(A, k=-k)
            B = torch.rot90(B, k=-k)
            C = torch.rot90(C, k=-k)
            D = torch.rot90(D, k=-k)

            output_A.append(A.cpu().numpy())
            output_B.append(B.cpu().numpy())
            output_C.append(C.cpu().numpy())
            output_D.append(D.cpu().numpy())

    output_A = np.mean(np.array(output_A), axis=0)
    output_B = np.mean(np.array(output_B), axis=0)
    output_C = np.mean(np.array(output_C), axis=0)
    output_D = np.mean(np.array(output_D), axis=0)

    if save:
        # map [-1, 1] back to [0, 65525]
        mapping = ScaleIntensityRange(a_min=-1, a_max=1, b_min=0, b_max=np.iinfo(np.uint16).max)
        output_A = mapping(output_A)
        output_B = mapping(output_B)
        output_C = mapping(output_C)
        output_D = mapping(output_D)

        if not osp.exists(osp.join(save_dir, 'mitochondria-fluorescence-ome-tiff')):
            os.mkdir(osp.join(save_dir, 'mitochondria-fluorescence-ome-tiff'))
        if not osp.exists(osp.join(save_dir, 'nucleus-fluorescence-ome-tiff')):
            os.mkdir(osp.join(save_dir, 'nucleus-fluorescence-ome-tiff'))
        if not osp.exists(osp.join(save_dir, 'tubulin-fluorescence-ome-tiff')):
            os.mkdir(osp.join(save_dir, 'tubulin-fluorescence-ome-tiff'))
        if not osp.exists(osp.join(save_dir, 'actin-fluorescence-ome-tiff')):
            os.mkdir(osp.join(save_dir, 'actin-fluorescence-ome-tiff'))

        save_image(osp.join(save_dir, 'mitochondria-fluorescence-ome-tiff', osp.basename(image_path)), output_A, metadata)
        save_image(osp.join(save_dir, 'nucleus-fluorescence-ome-tiff', osp.basename(image_path)), output_B, metadata)
        save_image(osp.join(save_dir, 'tubulin-fluorescence-ome-tiff', osp.basename(image_path)), output_C, metadata)
        save_image(osp.join(save_dir, 'actin-fluorescence-ome-tiff', osp.basename(image_path)), output_D, metadata)

    return output_A, output_B, output_C, output_D



def uni_synthesize(image_path: str, save_dir: str, save=True):  
    """BF: 0, PC: 1, DIC: 2"""
    opt = TestOptions().parse()
    opt.netG = 'res_multi_decoder'
    opt.norm = 'batch'

    with tifffile.TiffFile(image_path) as tif:
        metadata = tif.ome_metadata
    description = xmltodict.parse(metadata)
    tl= description["OME"]['Image']["Pixels"]["Channel"]["@Name"]
    opt.name = 'unified'  # unified model
    model = load_model(opt).netG
    data_dict = LightTransform()({'input': image_path})

    # Test-time augmentation
    output_A, output_B, output_C, output_D = [], [], [], [] 

    with torch.no_grad():
        for k in range(4): # customize from 1->4
            input_data = data_dict['input'].squeeze(0)
            # test-time augmentation: rotate
            input_data = torch.rot90(input_data, k=k).unsqueeze(0)
            # inference
            A, B, C, D = sliding_window_inference(
                inputs=input_data.unsqueeze(0).cuda(), 
                roi_size=(512, 512), 
                sw_batch_size=24, 
                predictor=model,
                overlap=0.8,
                mode='gaussian',
                sigma_scale=0.125,
                padding_mode='constant',
                cval=-1,
                is_train=False)[0]

            # test-time augmentation: rotate back
            A = torch.rot90(A, k=-k)
            B = torch.rot90(B, k=-k)
            C = torch.rot90(C, k=-k)
            D = torch.rot90(D, k=-k)

            output_A.append(A.cpu().numpy())
            output_B.append(B.cpu().numpy())
            output_C.append(C.cpu().numpy())
            output_D.append(D.cpu().numpy())

    output_A = np.mean(np.array(output_A), axis=0)
    output_B = np.mean(np.array(output_B), axis=0)
    output_C = np.mean(np.array(output_C), axis=0)
    output_D = np.mean(np.array(output_D), axis=0)

    if save:
        # map [-1, 1] back to [0, 65525]
        mapping = ScaleIntensityRange(a_min=-1, a_max=1, b_min=0, b_max=np.iinfo(np.uint16).max)
        output_A = mapping(output_A)
        output_B = mapping(output_B)
        output_C = mapping(output_C)
        output_D = mapping(output_D)

        if not osp.exists(osp.join(save_dir, 'mitochondria-fluorescence-ome-tiff')):
            os.mkdir(osp.join(save_dir, 'mitochondria-fluorescence-ome-tiff'))
        if not osp.exists(osp.join(save_dir, 'nucleus-fluorescence-ome-tiff')):
            os.mkdir(osp.join(save_dir, 'nucleus-fluorescence-ome-tiff'))
        if not osp.exists(osp.join(save_dir, 'tubulin-fluorescence-ome-tiff')):
            os.mkdir(osp.join(save_dir, 'tubulin-fluorescence-ome-tiff'))
        if not osp.exists(osp.join(save_dir, 'actin-fluorescence-ome-tiff')):
            os.mkdir(osp.join(save_dir, 'actin-fluorescence-ome-tiff'))

        save_image(osp.join(save_dir, 'mitochondria-fluorescence-ome-tiff', osp.basename(image_path)), output_A, metadata)
        save_image(osp.join(save_dir, 'nucleus-fluorescence-ome-tiff', osp.basename(image_path)), output_B, metadata)
        save_image(osp.join(save_dir, 'tubulin-fluorescence-ome-tiff', osp.basename(image_path)), output_C, metadata)
        save_image(osp.join(save_dir, 'actin-fluorescence-ome-tiff', osp.basename(image_path)), output_D, metadata)
    return output_A, output_B, output_C, output_D


def uni_dyn_synthesize(image_path: str, save_dir: str, save=True):  
    """BF: 0, PC: 1, DIC: 2"""
    opt = TestOptions().parse()
    opt.netG = 'res_dyn_multi_decoder'
    opt.norm = 'batch'

    with tifffile.TiffFile(image_path) as tif:
        metadata = tif.ome_metadata
    description = xmltodict.parse(metadata)
    tl= description["OME"]['Image']["Pixels"]["Channel"]["@Name"]
    opt.name = 'unified'  # unified model
    model = load_model(opt).netG
    data_dict = LightTransform()({'input': image_path})

    if tl == 'BF':
        data_dict['input_code'] = torch.tensor([0,0,1])
    elif tl == 'PC':
        data_dict['input_code'] = torch.tensor([1,0,0])
    elif tl == 'DIC':
        data_dict['input_code'] = torch.tensor([0,1,0])
    
    # Test-time augmentation
    output_A, output_B, output_C, output_D = [], [], [], [] 

    with torch.no_grad():
        for k in range(4): # customize from 1->4
            input_data = data_dict['input'].squeeze(0)
            # test-time augmentation: rotate
            input_data = torch.rot90(input_data, k=k).unsqueeze(0)
            # inference
            A, B, C, D = sliding_window_inference(
                inputs=input_data.unsqueeze(0).cuda(), 
                roi_size=(512, 512), 
                sw_batch_size=24, 
                predictor=model,
                overlap=0.8,
                mode='gaussian',
                sigma_scale=0.125,
                padding_mode='constant',
                cval=-1,
                is_train=False,
                code=data_dict['input_code'])[0]

            # test-time augmentation: rotate back
            A = torch.rot90(A, k=-k)
            B = torch.rot90(B, k=-k)
            C = torch.rot90(C, k=-k)
            D = torch.rot90(D, k=-k)

            output_A.append(A.cpu().numpy())
            output_B.append(B.cpu().numpy())
            output_C.append(C.cpu().numpy())
            output_D.append(D.cpu().numpy())

    output_A = np.mean(np.array(output_A), axis=0)
    output_B = np.mean(np.array(output_B), axis=0)
    output_C = np.mean(np.array(output_C), axis=0)
    output_D = np.mean(np.array(output_D), axis=0)

    if save:
        # map [-1, 1] back to [0, 65525]
        mapping = ScaleIntensityRange(a_min=-1, a_max=1, b_min=0, b_max=np.iinfo(np.uint16).max)
        output_A = mapping(output_A)
        output_B = mapping(output_B)
        output_C = mapping(output_C)
        output_D = mapping(output_D)

        if not osp.exists(osp.join(save_dir, 'mitochondria-fluorescence-ome-tiff')):
            os.mkdir(osp.join(save_dir, 'mitochondria-fluorescence-ome-tiff'))
        if not osp.exists(osp.join(save_dir, 'nucleus-fluorescence-ome-tiff')):
            os.mkdir(osp.join(save_dir, 'nucleus-fluorescence-ome-tiff'))
        if not osp.exists(osp.join(save_dir, 'tubulin-fluorescence-ome-tiff')):
            os.mkdir(osp.join(save_dir, 'tubulin-fluorescence-ome-tiff'))
        if not osp.exists(osp.join(save_dir, 'actin-fluorescence-ome-tiff')):
            os.mkdir(osp.join(save_dir, 'actin-fluorescence-ome-tiff'))

        save_image(osp.join(save_dir, 'mitochondria-fluorescence-ome-tiff', osp.basename(image_path)), output_A, metadata)
        save_image(osp.join(save_dir, 'nucleus-fluorescence-ome-tiff', osp.basename(image_path)), output_B, metadata)
        save_image(osp.join(save_dir, 'tubulin-fluorescence-ome-tiff', osp.basename(image_path)), output_C, metadata)
        save_image(osp.join(save_dir, 'actin-fluorescence-ome-tiff', osp.basename(image_path)), output_D, metadata)

    return output_A, output_B, output_C, output_D


def ensemble_synthesize(image_path: str, save_dir: str, save=True):
    start = time()
    print('============================== Start inference ==============================')
    output_A1, output_B1, output_C1, output_D1 = synthesize(image_path, save_dir, False)
    output_A2, output_B2, output_C2, output_D2 = uni_synthesize(image_path, save_dir, False)
    output_A3, output_B3, output_C3, output_D3 = uni_dyn_synthesize(image_path, save_dir, False)
    print(f'============================== Elapsed time: {time()-start:.1f}s ==============================')

    output_A = (output_A1 + output_A2 + output_A3) / 3
    output_B = (output_B1 + output_B2 + output_B3) / 3
    output_C = (output_C1 + output_C2 + output_C3) / 3
    output_D = (output_D1 + output_D2 + output_D3) / 3

    with tifffile.TiffFile(image_path) as tif:
        metadata = tif.ome_metadata
    description = xmltodict.parse(metadata)
    tl = description["OME"]['Image']["Pixels"]["Channel"]["@Name"]
    if tl == 'DIC':
        output_D = (output_D2 + output_D3) / 2

    if save:
        # map [-1, 1] back to [0, 65525]
        mapping = ScaleIntensityRange(a_min=-1, a_max=1, b_min=0, b_max=np.iinfo(np.uint16).max)
        output_A = mapping(output_A)
        output_B = mapping(output_B)
        output_C = mapping(output_C)
        output_D = mapping(output_D)

        if not osp.exists(osp.join(save_dir, 'mitochondria-fluorescence-ome-tiff')):
            os.mkdir(osp.join(save_dir, 'mitochondria-fluorescence-ome-tiff'))
        if not osp.exists(osp.join(save_dir, 'nucleus-fluorescence-ome-tiff')):
            os.mkdir(osp.join(save_dir, 'nucleus-fluorescence-ome-tiff'))
        if not osp.exists(osp.join(save_dir, 'tubulin-fluorescence-ome-tiff')):
            os.mkdir(osp.join(save_dir, 'tubulin-fluorescence-ome-tiff'))
        if not osp.exists(osp.join(save_dir, 'actin-fluorescence-ome-tiff')):
            os.mkdir(osp.join(save_dir, 'actin-fluorescence-ome-tiff'))

        save_image(osp.join(save_dir, 'mitochondria-fluorescence-ome-tiff', osp.basename(image_path)), output_A, metadata)
        save_image(osp.join(save_dir, 'nucleus-fluorescence-ome-tiff', osp.basename(image_path)), output_B, metadata)
        save_image(osp.join(save_dir, 'tubulin-fluorescence-ome-tiff', osp.basename(image_path)), output_C, metadata)
        save_image(osp.join(save_dir, 'actin-fluorescence-ome-tiff', osp.basename(image_path)), output_D, metadata)

    return output_A, output_B, output_C, output_D


# validation
if __name__ == '__main__':
    synthesize('/data/LightUp/docker_submission/docker_template_algorithm/0408/input/images/organelles-transmitted-light-ome-tiff/image_158_PC_z0.ome.tiff', '/data/LightUp/results/temp')
    # uni_synthesize('/data/LightUp/Database/Study_21/image_321_DIC_z7.ome.tiff', '/data/LightUp/results/temp')
    # uni_dyn_synthesize('/data/LightUp/docker_submission/docker_template_algorithm/0408/input/images/organelles-transmitted-light-ome-tiff/image_158_PC_z0.ome.tiff', '/data/LightUp/results/temp')
    # ensemble_synthesize('/data/LightUp/Database/Study_21/image_321_DIC_z7.ome.tiff', '/data/LightUp/results/temp')
    
                
