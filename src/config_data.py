'''
    config constants for training neural net  
'''
# the remainder is allocated as test data
PERCENT_FOR_TRAINING_DATA = .98

# the patch size for both the 32 and 64 feature convolutions
# used with an NxN tile, where N has usually been 64
CONVOLUTION_PATCH_SIZE = 5
