import json
from os import listdir
from os.path import isfile, join, isdir, exists

from pyparsing import Dict
from torchvision.io import decode_image
from torch.utils.data import Dataset
from pathlib import Path
from collections import defaultdict
import torch

class TUMTraf(Dataset):

    CLASSES = ['BICYCLE', 
               'BUS', 
               'CAR', 
            #   'MOTORCYCLE', 
               'PEDESTRIAN', 
               'TRAILER', 
               'TRUCK']

    def __init__(self, img_dir:Path, label_dir:Path, by_group:bool=False):
        """
        Assumes data is on groups, wont work if doesnt run preprocessed/
        if data is loaded by group, each item is a set of frames, else every item is a frame
        
        :param img_dir: Path where preprocessed images are stored
        :type img_dir: Path
        :param label_dir: Path where preprocessed labels are stored
        :type label_dir: Path
        :param by_group: Whether to load data by group
        :type by_group: bool
        """
        # TODO check when data structure is not the same, or not using preprocessed bc if not it just quietly fails
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.by_group = by_group
        self.classes = self.CLASSES
    
        if not isdir(self.img_dir) or not exists(self.img_dir):
            raise ValueError("invalid img_dir", self.img_dir)

        if not isdir(self.label_dir) or not exists(self.label_dir):
            raise ValueError("invalid label_dir", self.label_dir)
        

        self.img_labels = defaultdict(dict)
        # # example dict {'openlabel': {'metadata': {'schema_version': '1.0.0'}, 'coordinate_systems': 169, 'frames': {'169': {'objects': {'0': {'object_data': {'name': 'PEDESTRIAN_0', 'type': 'PEDESTRIAN', 'bbox': [{'name': 'full_bbox', 'val': [568, 167, 38, 58], 'attributes': {'text': [{'name': 'sensor_id', 'val': 'default_cam'}]}}]}}, '1': {'object_data': {'name': 'PEDESTRIAN_1', 'type': 'PEDESTRIAN', 'bbox': [{'name': 'full_bbox', 'val': [574, 241, 46, 87], 'attributes': {'text': [{'name': 'sensor_id', 'val': 'default_cam'}]}}]}}, '2': {'object_data': {'name': 'CAR_2', 'type': 'CAR', 'bbox': [{'name': 'full_bbox', 'val': [258, 24, 46, 39], 'attributes': {'text': [{'name': 'sensor_id', 'val': 'default_cam'}]}}]}}, '3': {'object_data': {'name': 'TRUCK_3', 'type': 'TRUCK', 'bbox': [{'name': 'full_bbox', 'val': [309, 137, 81, 135], 'attributes': {'text': [{'name': 'sensor_id', 'val': 'default_cam'}]}}]}}, '4': {'object_data': {'name': 'CAR_4', 'type': 'CAR', 'bbox': [{'name': 'full_bbox', 'val': [392, 22, 34, 36], 'attributes': {'text': [{'name': 'sensor_id', 'val': 'default_cam'}]}}]}}, '5': {'object_data': {'name': 'TRAILER_5', 'type': 'TRAILER', 'bbox': [{'name': 'full_bbox', 'val': [347, 28, 48, 56], 'attributes': {'text': [{'name': 'sensor_id', 'val': 'default_cam'}]}}]}}, '6': {'object_data': {'name': 'CAR_6', 'type': 'CAR', 'bbox': [{'name': 'full_bbox', 'val': [262, 33, 45, 43], 'attributes': {'text': [{'name': 'sensor_id', 'val': 'default_cam'}]}}]}}, '7': {'object_data': {'name': 'CAR_7', 'type': 'CAR', 'bbox': [{'name': 'full_bbox', 'val': [375, 40, 41, 43], 'attributes': {'text': [{'name': 'sensor_id', 'val': 'default_cam'}]}}]}}, '8': {'object_data': {'name': 'TRAILER_8', 'type': 'TRAILER', 'bbox': [{'name': 'full_bbox', 'val': [126, 107, 27, 166], 'attributes': {'text': [{'name': 'sensor_id', 'val': 'default_cam'}]}}]}}, '9': {'object_data': {'name': 'CAR_9', 'type': 'CAR', 'bbox': [{'name': 'full_bbox', 'val': [195, 49, 56, 49], 'attributes': {'text': [{'name': 'sensor_id', 'val': 'default_cam'}]}}]}}}}}}}
        group_folders = [f for f in listdir(self.label_dir) 
                            if not isfile(join(self.label_dir, f))]
        for group in group_folders:
            group_path = join(self.label_dir, group)
            onlyfiles = [f for f in listdir(group_path) 
                            if isfile(join(group_path, f))]
            for i,file in enumerate(onlyfiles):
                with open(f"{group_path}/{file}") as f:
                    d = json.load(f)
                    file_id = file.split(".json")[0]
                    if self.by_group:
                        self.img_labels[group][file_id] = d
                    else:
                        self.img_labels[file_id] = {'group': group, 'data': d}

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):

        # TODO consider to change idx to by index and not literal group id?

        if self.by_group:
            # returns all frames in the group of idx

            return [{"frame":decode_image(f"{self.img_dir}/{idx}/{key}.jpg"),"label":item} 
                    for key,item in self.img_labels[idx].items()]
        image = decode_image(f"{self.img_dir}/{self.img_labels[idx]['group']}/{self.img_labels[idx]['id']}.jpg")
        return {"frame":image,"label":self.img_labels[idx]['data']}
    


class TUMTrafSSD(Dataset):
    """
    Dataset wrapper for TUMTraf that formats data for SSD training.
    Loads individual frames (not by group) for ANN training.
    """
    
    CLASSES = ['BICYCLE', 'BUS', 'CAR', 'PEDESTRIAN', 'TRAILER', 'TRUCK']
    
    def __init__(self, img_dir: Path, label_dir: Path, transform=None, target_size: tuple[int, int] = (480, 640)):
        """
        Args:
            img_dir: Path to preprocessed images (e.g., data/preprocessed/train/images/rgb)
            label_dir: Path to preprocessed labels (e.g., data/preprocessed/train/OPENLabel_labels_rgb)
            transform: Optional transforms to apply to images
            target_size: (height, width) to resize images
        """
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.target_size = target_size
        self.class_to_idx = {cls: idx + 1 for idx, cls in enumerate(self.CLASSES)}  # 1-indexed (0 is background)
        
        # Collect all image-label pairs
        self.samples = []
        
        # Iterate through group folders
        for group_folder in sorted(self.label_dir.iterdir()):
            if not group_folder.is_dir():
                continue
                
            for label_file in sorted(group_folder.glob("*.json")):
                file_id = label_file.stem
                img_path = self.img_dir / group_folder.name / f"{file_id}.jpg"
                
                if img_path.exists():
                    self.samples.append({
                        'img_path': img_path,
                        'label_path': label_file,
                        'group': group_folder.name,
                        'file_id': file_id
                    })
        
        print(f"Loaded {len(self.samples)} samples from {img_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        sample = self.samples[idx]
        
        # Load image
        image = decode_image(str(sample['img_path']))  # (C, H, W)
        
        # Get original size for bbox normalization
        _, orig_h, orig_w = image.shape
        
        # Load labels
        with open(sample['label_path'], 'r') as f:
            label_data = json.load(f)
        
        # Parse bounding boxes and classes
        boxes = []
        labels = []
        
        frames = label_data.get('openlabel', {}).get('frames', {})
        for frame_id, frame_content in frames.items():
            objects = frame_content.get('objects', {})
            for obj_id, obj_content in objects.items():
                obj_data = obj_content.get('object_data', {})
                obj_type = obj_data.get('type', '')
                
                # Skip classes not in our list
                if obj_type not in self.class_to_idx:
                    continue
                
                bbox_list = obj_data.get('bbox', [])
                for bbox in bbox_list:
                    if bbox.get('name') == 'full_bbox':
                        # bbox format: [x, y, width, height]
                        x, y, w, h = bbox.get('val', [0, 0, 0, 0])
                        
                        # Convert to normalized (cx, cy, w, h)
                        cx = (x + w / 2) / orig_w
                        cy = (y + h / 2) / orig_h
                        nw = w / orig_w
                        nh = h / orig_h
                        
                        # Skip invalid boxes
                        if nw > 0 and nh > 0:
                            boxes.append([cx, cy, nw, nh])
                            labels.append(self.class_to_idx[obj_type])
        
        # Convert to tensors
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default normalization
            image = image.float() / 255.0
        
        return image, {'boxes': boxes, 'labels': labels}
