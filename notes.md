about the data:
- group in a reasonable amount of n_frames and max_diff. note on the 'reasonable'
    - check later what is a good amount. for now 8-16 frames and 1s (1000ms) seems alright
- crop with the eb_roi_text in the calibration folders. a little bit of data gets lost compared to mine but is fine for consistency with the paper
- groupids dont match, group 0003 of rgb mihht not matb group 0003 eb_transformed
- there are some images that cleary show a car and are not labeled. should i label? https://www.perplexity.ai/search/i-am-seeing-my-datasets-tumtra-c0aN71xFSVy_VsY1Q1YNFg
- imbalance on classes
    RGB train classes: ['BICYCLE', 'BUS', 'CAR', 'PEDESTRIAN', 'TRAILER', 'TRUCK']
    EB train classes: ['BICYCLE', 'BUS', 'CAR', 'MOTORCYCLE', 'PEDESTRIAN', 'TRAILER', 'TRUCK']
    RGB val classes: ['BUS', 'CAR', 'MOTORCYCLE', 'PEDESTRIAN', 'TRAILER', 'TRUCK']
    EB val classes: ['BUS', 'CAR', 'PEDESTRIAN', 'TRAILER', 'TRUCK']
    RGB test day classes: ['BICYCLE', 'BUS', 'CAR', 'MOTORCYCLE', 'PEDESTRIAN', 'TRAILER', 'TRUCK']
    EB test day classes: ['BICYCLE', 'BUS', 'CAR', 'MOTORCYCLE', 'PEDESTRIAN', 'TRAILER', 'TRUCK']
    RGB test night_with_light_off classes: ['CAR']
    EB test night_with_light_off classes: ['CAR']
    RGB test night_with_light_on classes: ['BUS', 'CAR', 'TRUCK']
    EB test night_with_light_on classes: ['BUS', 'CAR', 'TRUCK']

    removing the class for consistency
- not super 1 to 1, eb lacks a lot of annotation
        Classes: ['BICYCLE', 'BUS', 'CAR', 'MOTORCYCLE', 'PEDESTRIAN', 'TRAILER', 'TRUCK']
        Total Samples: 2360
        Empty Samples (after filtering): 0 (0.00%)
        Class Distribution:
        CAR: 8104
        TRUCK: 706
        PEDESTRIAN: 694
        TRAILER: 640
        BUS: 337
        BICYCLE: 17
        MOTORCYCLE: 2

        Classes: ['BICYCLE', 'BUS', 'CAR', 'MOTORCYCLE', 'PEDESTRIAN', 'TRAILER', 'TRUCK']
        Total Samples: 2358
        Empty Samples (after filtering): 0 (0.00%)
        Class Distribution:
        CAR: 17499
        TRUCK: 2052
        PEDESTRIAN: 1237
        TRAILER: 1004
        BUS: 414
        BICYCLE: 47
    there is too few matching groups, proceeding as is for now
    Total Objects RGB Train: 22253
Total Objects EB Train: 10498
Ratio: 0.47

Perfect Matches: 0 groups
Close Matches (<10% diff): 6 groups
Mismatches: 289 groups

Top 5 Mismatches (Group, RGB Train, EB Train):
  ('0084', 126, 20)
  ('0170', 115, 12)
  ('0085', 118, 16)
  ('0082', 118, 19)
  ('0083', 118, 20)


for rgb:
- in training: dont care about videos, just annotated images
- in evaluation: care about videos

for evt:
- in training: care about videos
- in evaluation: care about video
- look for coding method (population etc )


----

if i remove N pixels, which shape or distribution D around the image removes more information?
eg. i would say a diagonal from left corner to right down corner removes the most because thats the intersection where most cars pass

rgb and eb being different size hinders performance fundamentally?

what fundamental difference about information content there is between both formats (rgb and eb)


TODO:
    - [] check if i need to transform the format of the eb images with tonic


---

change to this project structure (template)
project_name/
├── README.md               # Instructions to reproduce results
├── requirements.txt        # Dependencies
├── run.sh                  # Main entry point script
│
├── data/                   # Data directory (often added to .gitignore)
│   ├── raw/                # Immutable original data (e.g., TUMTraf_Event_Dataset)
│   ├── processed/          # The data after running preprocess.py
│   └── external/           # Data from third party sources
│
├── notebooks/              # Jupyter notebooks for exploration (e.g., peekabo.ipynb)
│   ├── exploratory/
│   └── paper_figures/      # Notebooks specifically for generating paper plots
│
├── src/                    # Source code (your python package)
│   ├── __init__.py
│   ├── data/               # Data handling
│   │   ├── __init__.py
│   │   ├── dataset.py      # PyTorch Dataset class (TUMTraf)
│   │   └── transforms.py   # Data augmentations/transformations
│   │
│   ├── models/             # Model definitions
│   │   ├── __init__.py
│   │   ├── snn_model.py    # Your SNN architecture
│   │   └── ann_model.py    # Your ANN architecture
│   │
│   └── utils/              # Helper functions (metrics, logging, etc.)
│
├── scripts/                # Scripts to run pipelines (or keep in root if few)
│   ├── train.py            # Training loop (merging train_ann_rgb.py & train_snn_eb.py)
│   ├── evaluate.py         # Evaluation on test set
│   └── preprocess.py       # Script to turn raw -> processed
│
└── results/                # Artifacts
    ├── checkpoints/        # Saved model weights (.pt/.pth)
    ├── logs/               # Tensorboard or CSV logs
    └── figures/            # Generated plots for the paper
---

TODO Lategame
    - [] compute shannon entropy by regions by hand
    - [] check why roi is wrong. using it but shifting by multiplying by 10 to keep it consistent
    - [] use fred https://miccunifi.github.io/FRED/ 