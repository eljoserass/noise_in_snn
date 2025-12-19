about the data:
- group in a reasonable amount of n_frames and max_diff. note on the 'reasonable'
    - check later what is a good amount. for now 8-16 frames and 1s (1000ms) seems alright
- crop with the eb_roi_text in the calibration folders. a little bit of data gets lost compared to mine but is fine for consistency with the paper
- groupids dont match, group 0003 of rgb mihht not matb group 0003 eb_transformed
- there are some images that cleary show a car and are not labeled. should i label? https://www.perplexity.ai/search/i-am-seeing-my-datasets-tumtra-c0aN71xFSVy_VsY1Q1YNFg
 


for rgb:
- in training: dont care about videos, just annotated images
- in evaluation: care about videos

for evt:
- in training: care about videos
- in evaluation: care about video
- look for coding method (population etc )


----

rgb and eb being different size hinders performance fundamentally?

what fundamental difference about information content there is between both formats (rgb and eb)


TODO:
    - [] adapat torch dataset to use the new structure
    - [] check if i need to transform the format of the eb images with tonic


TODO Lategame
    - [] compute shannon entropy by regions by hand
    - [] check why roi is wrong. using it but shifting by multiplying by 10 to keep it consistent
    - [] use fred https://miccunifi.github.io/FRED/ 