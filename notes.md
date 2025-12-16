about the data:
- the left strip is on eb_Transformed==== more on how to solve it here https://www.perplexity.ai/search/what-are-the-calibraations-fol-YgwXHY21RM2AaZJh3euGKw
    https://github.com/tum-traffic-dataset/tum-traffic-dataset-dev-kit/blob/a11c99b5b7f80b5fd1d8a2c9398fd6d21b4c80e2/src/preprocessing/undistort_images.py
    undisort opencv
- group the frames by seconds or something, pick the frames


for rgb:
- in training: dont care about videos, just annotated images
- in evaluation: care about videos

for evt:
- in training: care about videos
- in evaluation: care about video
- look for coding method (population etc )


paths to see where there are no events
![alt text](data/TUMTraf_Event_Dataset/val/images/eb_transformed/20231114-082225.660894.jpg)
- corner that can be used for left and bottom


![alt text](data/TUMTraf_Event_Dataset/val/images/eb_transformed/20231114-082301.647477.jpg)
- can be used for right side


![alt text](data/TUMTraf_Event_Dataset/val/images/eb_transformed/20231114-082434.166224.jpg)
- good for corners

![alt text](data/TUMTraf_Event_Dataset/val/images/eb_transformed/20231114-083734.164714.jpg)
- good for left


find the tiles of dead pixels, lets say that im adding an extra filtering step on the preprocessing pipeline?   