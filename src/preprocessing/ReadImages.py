from os import listdir
from os.path import join
from aicsimageio.readers.bioformats_reader import BioFile



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


# Create Image files dictionary  
def get_dict(db_path=db_path):
    images={}
    for study in listdir(db_path):
        print(" add images from "+study)
        for filename in listdir(join(db_path,study)):
            number=get_image_number(filename)
            if number  not in images :  images[number]={}

            if is_input_image(filename):
                if 'input' not in images[number]: images[number]['input']={}
                images[number]['input'][get_z(filename)]=join(db_path,study,filename)
            else: 
                if 'output' not in images[number]: images[number]['output']={}
                images[number]['output'][get_channel_type(filename)]=join(db_path,study,filename)
    return images


# numbers = []

# for i in range(len(images)):
#     numbers.append(len(images[i]['output']))

# import numpy as np
# np.unique(numbers)

# breakpoint()

#Read each individual Images
#Requirement : https://pypi.org/project/aicsimageio/

x, y = [], []

for number in images:
    print(" Read Images number "+str(number))
    for image_type in images[number]:#input, output
        for w in images[number][image_type]: #Can be z value for input or channels for output 
            im=BioFile(images[number][image_type][w])
            continue
            # print(images[number][image_type][w]+ " as shape "+ str(im.to_numpy().shape))
        print('hi')
        x.append(im.to_numpy().shape[-2])
        y.append(im.to_numpy().shape[-1])
        continue
        

# (Pdb) np.unique(x, return_counts=True)
# (array([ 512,  966,  980, 1024, 1200, 1300, 2044, 2048]), array([2578,   62, 1174,   20,  322,   60,  112,  820]))
# (Pdb) np.unique(y, return_counts=True)
# (array([ 512, 1016, 1024, 1200, 1296, 1624, 2048]), array([2578, 1174,   20,  322,   62,   60,  932]))

breakpoint()