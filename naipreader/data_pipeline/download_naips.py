'''
    a class to download NAIP imagery from the aws-naip
    pay-as-you-go bucket and process it

'''

import sys, os
import boto3

class NAIPDownloader:  

  def __init__(self):
  	self.make_directory('data/naip/', full_path=True)

  def make_directory(self, new_dir, full_path=False):
    '''
       try to make a new directory
    '''
    if full_path:
      path = ''
      for token in new_dir.split('/'):
        path += token + '/'
        print (path)
        try:
          print ("try")
          os.mkdir(path);
        except:
          print ("Except")
          pass
      return path

    try:
      os.mkdir(new_dir);
    except:
      pass
    return new_dir

  def download_naips(self):
    s3_client = boto3.client('s3')
    filename = 'm_3807708_ne_18_1_20130924.tif'
    full_path = 'data/naip/{}'.format(filename)
    if os.exists(full_path):
      print("{} already downloaded".format(full_path))
    else:
      s3_client.download_file('aws-naip', 'md/2013/1m/rgbir/38077/{}'.format(filename), full_path, {'RequestPayer':'requester'})
parameters_message = "parameters are: download"
if len(sys.argv) == 1:
  print(parameters_message)
elif sys.argv[1] == 'download':
	naiper = NAIPDownloader()
	naiper.download_naips()
else:
  print(parameters_message)

