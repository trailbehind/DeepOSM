'''
    a class to download NAIP imagery from the aws-naip
    pay-as-you-go bucket and process it

'''

import sys
import boto3
from marshall_osm_data import OSMDataNormalizer

class NAIPDownloader:  

  def __init__(self):
  	odn = OSMDataNormalizer('')
  	odn.make_directory('data/naip/')

  def download_naips(self):
    s3_client = boto3.client('s3')
    filename = 'm_3807708_ne_18_1_20130924.tif'
    s3_client.download_file('aws-naip', 'md/2013/1m/rgbir/38077/{}'.format(filename), 'data/naip/{}'.format(filename), {'RequestPayer':'requester'})

parameters_message = "parameters are: download"
if len(sys.argv) == 1:
  print(parameters_message)
elif sys.argv[1] == 'download':
	naiper = NAIPDownloader()
	naiper.download_naips()
else:
  print(parameters_message)

