# this will prepare the rgb and eb data. 
# it will use read data from train/test/val
# it will strip the lines that are empty from eb_transformed
# it will write write the labels for eb_transformed with the labels adjusted is probably easier
# it will group the frames into little videos of 1-2 s.
#   5-7 fps = 8-16 frames per video.
#   checking ms difference between frames, if its to large and we couldnt group. drop and try to group again
#   we will write it in a json or in folders idk


"""

i need to find where is the left strip, and generate a json of the packets of frames
then simply either use that index to return the data propperly on the dataset torch wrapper or pre-compute the transformation, but with the cost of doubling the storage pretty much

is better still because is something that ill do once, and space is cheaper than computation.

option b is better: create preprocessed fodler or something that removes the left strip, shift labels, and packets frames in in groups
this for train/val/test. group the same for rgb to be fair with how many data and which data we use. then simply train on frames

""" 