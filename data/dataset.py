import json
from os import listdir
from os.path import isfile, join, isdir, exists
from torchvision.io import decode_image
from torch.utils.data import Dataset

class TUMTraf(Dataset):
    def __init__(self, img_dir, label_dir,
                transform=None, target_transform=None):
        """
        Data loader for the rgb split of the TUMTRAF
        argrs:
            - img_dir = where the images are located, here include if rgb or eb
                the structure should be virgin from the zip
            - label_dir = where the OPENLabel annotation are
                the subset of images and annoation must match 
                the path will also contain the split and subset
            - transform and target_transform idk
        """
        self.transform = transform
        self.target_transform = target_transform
        self.img_dir = img_dir
        self.label_dir = label_dir

        self.img_labels = []

        # TODO check that subset annotations and data match and rgb vs event
        if not isdir(self.img_dir) or not exists(self.img_dir):
            raise ValueError("invalid image_dir")

        if not isdir(self.label_dir) or not exists(self.label_dir):
            raise ValueError("invalid label_dir")
       
        onlyfiles = [f for f in listdir(self.label_dir) 
                     if isfile(join(self.label_dir, f))]
        for file in onlyfiles:
            with open(f"{self.label_dir}/{file}") as f:
                d = json.load(f)
                file_id = file.split(".json")[0]
                self.img_labels.append({"id":file_id, "data": d})


    def __len__(self):
        return len(self.img_labels)
    def __getitem__(self, idx):
        image = decode_image(f"{self.img_dir}/{self.img_labels[idx]['id']}.jpg")
        return image,self.img_labels[idx]