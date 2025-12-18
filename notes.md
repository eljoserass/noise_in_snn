about the data:
- group in a reasonable amount of n_frames and max_diff. note on the 'reasonable'
    - check later what is a good amount. for now 8-16 frames and 1s (1000ms) seems alright
- crop with the eb_roi_text in the calibration folders. a little bit of data gets lost compared to mine but is fine for consistency with the paper
 


for rgb:
- in training: dont care about videos, just annotated images
- in evaluation: care about videos

for evt:
- in training: care about videos
- in evaluation: care about video
- look for coding method (population etc )


----

do i need to apply the roi to the rgb?

TODO:
    - [] apply eb_roi to preprocessed eb data (images and labels)
    - [] adapat torch dataset to use the new structure
    - [] check if i need to transform the format of the eb images
    - [] create a cli for a option of the pipeline to visualize videos and/or painted frames


TODO Lategame
    - [] check why roi is wrong. using it but shifting by multiplying by 10 to keep it consistent
    - [] use fred https://miccunifi.github.io/FRED/ 
