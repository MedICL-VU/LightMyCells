import os
import os.path as osp
from synthesize import synthesize


# Create new directories
INPUT_PATH = '/input/'
OUTPUT_PATH = '/output/'

# INPUT_PATH = '/data/LightUp/results/input'
# OUTPUT_PATH = '/data/LightUp/results/output'


if not osp.isdir(osp.join(INPUT_PATH)): os.mkdir(osp.join(INPUT_PATH))
if not osp.isdir(osp.join(OUTPUT_PATH)): os.mkdir(osp.join(OUTPUT_PATH))
if not osp.isdir(osp.join(OUTPUT_PATH,"images")): os.mkdir(osp.join(OUTPUT_PATH,"images"))


def run():
    transmitted_light_path = osp.join(INPUT_PATH , "images", "organelles-transmitted-light-ome-tiff")
    for input_file_name in os.listdir(transmitted_light_path):
        if input_file_name.endswith(".tiff"):
            print(" --> Predict " + input_file_name)
            synthesize(osp.join(transmitted_light_path, input_file_name), osp.join(OUTPUT_PATH, "images"))


if __name__ == "__main__":
    raise SystemExit(run())
