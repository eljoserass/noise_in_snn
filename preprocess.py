# this will prepare the rgb and eb data. 
# it will use read data from train/test/val
# it will strip the lines that are empty from eb_transformed
# it will write write the labels for eb_transformed with the labels adjusted is probably easier
# it will group the frames into little videos of 1-2 s.
#   5-7 fps = 8-16 frames per video.
#   checking ms difference between frames, if its to large and we couldnt group. drop and try to group again
#   we will write it in a json or in folders idk


#params
# --data-path <path>- where the root is - has default
# --out-path <path> - where to write the data output, if it exists something dont rewrite - has default
# --rewrite - rewrite content ignore if it exists - defaults to false
# --rgb - do it only on rgb split - defaults to false
# --eb - do it only on eb_transformed split - defaults to false
# --split <split,splita,...> - do it only in the specified split - defaults to train/val/test
# --eb_roi_path - where the json is - defaults to roi_eb_transformed.json
# --n_frames - number of frames per video -  defaults to 16

# returns
# if main
# will dump a folder with similar structure,
#  but each grup of frames in a different folder, and eb_transformed with new labels and image dimensions