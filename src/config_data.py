'''
    config choosing data and training the neural net
'''


'''
    constants for how to create labels, 
    from OpenStreetMap way (road) info in PBF files
'''
# enough to cover NAIPs around DC/Maryland/Virginia
PBF_FILE_URLS = ['http://download.geofabrik.de/north-america/us/maryland-latest.osm.pbf',
                 'http://download.geofabrik.de/north-america/us/virginia-latest.osm.pbf',
                 'http://download.geofabrik.de/north-america/us/district-of-columbia-latest.osm.pbf']

# tile the NAIP and training data into NxN tiles with this dimension
TILE_SIZE = 64

# the number of pixels to count as road, 
# on each side of of the centerline pixels
PIXELS_BESIDE_WAYS = 1

# to count an NxN tile as being "On" for roads,
# N*.25 pixels on that tiles must have been classified as roads
PERCENT_OF_TILE_HEIGHT_TO_ACTIVATE = .50

'''
    constants for NAIP imagery to use   
'''
# the bands to use from the NAIP for analysis (R G B IR)
BANDS_TO_USE = [1,1,1,1]

# set this to None to get different tifs to analyze
HARDCODED_NAIP_LIST = [
                  'm_3807708_ne_18_1_20130924.tif',
                  #'m_3807708_nw_18_1_20130904.tif',
                  #'m_3807708_se_18_1_20130924.tif',
                  #'m_3807708_se_18_1_20130924.tif'
                  ]

# values to create the S3 bucket path for some maryland NAIPs
# you can get random NAIPS from here, or the exact HARDCODED_NAIP_LIST above
# \todo document how to configure some of these
NAIP_STATE = 'md'
NAIP_YEAR = '2013'
NAIP_RESOLUTION = '1m'
NAIP_SPECTRUM = 'rgbir' 
NAIP_GRID = '38077'

# set this to a value between 1 and 10 or so,
# and unset HARDCODED_NAIP_LIST, to get some different NAIPs
NUMBER_OF_NAIPS = -1

# set this to True for production data science, False for debugging infrastructure
# speeds up downloads and matrix making when False
RANDOMIZE_NAIPS = False

'''
    constants for training neural net  
'''
# the remainder is allocated as test data
PERCENT_FOR_TRAINING_DATA = .93

# the number of batches to train the neural net
# @lacker recommends 3-5K for statistical significance, as rule of thumb
# can achieve 70% accuracy with 5000 so far
NUMBER_OF_BATCHES = 100

# the number of tiles for each training round
BATCH_SIZE = 100

# the patch size for both the 32 and 64 feature convolutions
# used with an NxN tile, where N has usually been 64
CONVOLUTION_PATCH_SIZE = 5


