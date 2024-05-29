import os
from aicsimageio.readers.bioformats_reader import BioFile



data_path = "/data/LightUp/Database"
url="https://seafile.lirmm.fr/d/123f71e12bf24db59d84/files/?p=%2F"

for s in range(1,31):
    study="Study_" + str(s)
    print("--> download "+study)
    os.system('wget "'+url+study+'.tar.gz&dl=1" --output-document='+os.path.join(data_path,study+".tar.gz"))
    os.system("cd "+data_path+"; tar -xf "+study+".tar.gz")
    for filename in os.listdir(os.path.join(data_path,study)):
        print(" -----> read "+os.path.join(study,filename))
        im = BioFile(os.path.join(data_path,study,filename))
        print(study+ " -> image " + filename+ " as shape " + str(im.to_numpy().shape) + " and metadata "+im.ome_xml)
