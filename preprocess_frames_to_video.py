


# choose the set defined by the path 
    # it can be day,night, train etc

# check that we are only using eb_transformed and rgb

# apply the undisort, possibly write new labebls that take that into account?


# split in --arg frames 
#   good range is aroud the 8-16 ms range http://perplexity.ai/search/where-the-literature-stands-on-qPXqQsOdRiGyUui_oCsHMA
#   check that by doing so there are not big jumps btween frames. check in ms
#   can consider also splitting on --arg seconds,ms and give the frames corresponding to that resolution


# group the filenames into videos depending on the criteria

# choose if we are going to write a directory for each video, or write a json that we can then use for the __getitem__ function?