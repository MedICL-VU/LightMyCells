import os
from os import listdir
from os.path import join
from aicsimageio.readers.bioformats_reader import BioFile
from tqdm import tqdm
import numpy as np
from glob import glob


db_path="/data/LightUp/Database" # Main Folder


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


# Create Image files dictionnary  
# images={}
# for study in listdir(db_path):
#     print(" add images from "+study)
#     for filename in listdir(join(db_path,study)):
#         number=get_image_number(filename)
#         if number  not in images :  
#             images[number]={}

#         if is_input_image(filename):
#             if 'input' not in images[number]: images[number]['input']={}
#             images[number]['input'][get_z(filename)]=join(db_path,study,filename)
#         else: 
#             if 'output' not in images[number]: images[number]['output']={}
#             images[number]['output'][get_channel_type(filename)]=join(db_path,study,filename)


# #Read each individual Images
# #Requirement : https://pypi.org/project/aicsimageio/

# save_dir = '/data/LightUp/Database/npy_data'

# with tqdm(total=len(images)) as pbar:
#     for number in images:
#         # print(" Read Images number "+str(number))
#         for image_type in images[number]:#input, output
#             for w in images[number][image_type]: #Can be z value for input or channels for output 
#                 filename = images[number][image_type][w]
#                 im=BioFile(filename).to_numpy()[0,0,0]
#                 # print(im.shape)
#                 np.save(join(save_dir, os.path.basename(filename).replace('.ome.tiff', '')), im)
#                 # print(images[number][image_type][w]+ " as shape "+ str(im.to_numpy().shape))


#         pbar.update(1)



paths = glob('/data/LightUp/Database/Study_17' + '/*.*')
stats = []

for path in paths:
    # if 'Mitochondria' in path:
    # if 'Nucleus' in path:
    # if 'Actin' in path:
    # if 'Tubulin' in path:
    # if 'PC' in path:
    # if 'BF' in path:
    if 'DIC' in path:
        im = BioFile(path).to_numpy()[0,0,0]
        high_p = np.percentile(im, 99.8)
        print(path, high_p)
        stats.append(high_p)

print('mean: ', np.mean(stats), np.median(stats), np.max(stats), np.percentile(stats, 99))