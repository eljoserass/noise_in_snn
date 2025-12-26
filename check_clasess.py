from src.data.dataset import TUMTraf
from pathlib import Path
import os

project_root = Path(os.path.dirname(os.path.abspath('__file__')))

data_dir_train = project_root / "data" / "preprocessed" / "train"

rgb_train_data = str(data_dir_train / "images" / "rgb")
eb_train_data = str(data_dir_train / "images" / "eb_transformed")

rgb_train_label = str(data_dir_train / "OPENLabel_labels_rgb")
eb_train_label = str(data_dir_train / "OPENLabel_labels_eb")

rgb_train_dataset = TUMTraf(
    img_dir=rgb_train_data,
    label_dir=rgb_train_label,
    by_group=False,
)

eb_train_dataset = TUMTraf(
    img_dir=eb_train_data,
    label_dir=eb_train_label,
    by_group=True,
)

print("RGB train classes:", rgb_train_dataset.classes)
print("EB train classes:", eb_train_dataset.classes)

print ("len rgb dataset:", len(rgb_train_dataset))
print ("len eb dataset:", len(eb_train_dataset))