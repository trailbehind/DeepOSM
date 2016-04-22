'''
    a class to download NAIP imagery from the aws-naip
    pay-as-you-go bucket and process it

'''

import sys, os
import boto3

GEO_DATA_DIR = os.environ.get("GEO_DATA_DIR") # set in Dockerfile as env variable
NAIP_DATA_DIR = os.path.join(GEO_DATA_DIR, "naip")

class NAIPDownloader:

  def __init__(self):
  	self.make_directory(NAIP_DATA_DIR, full_path=True)

  def make_directory(self, new_dir, full_path=False):
    '''
       try to make a new directory
    '''
    if full_path:
      path = ''
      for token in new_dir.split('/'):
        path += token + '/'
        try:
          os.mkdir(path);
        except:
          pass
      return path

    try:
      os.mkdir(new_dir);
    except:
      pass
    return new_dir

  def download_naips(self):
    # Return the full path downloaded to.
    s3_client = boto3.client('s3')
    filename = 'm_3807708_ne_18_1_20130924.tif'
    full_path = os.path.join(NAIP_DATA_DIR, filename)
    if os.path.exists(full_path):
      print("NAIP {} already downloaded".format(full_path))
    else:
      s3_client.download_file('aws-naip', 'md/2013/1m/rgbir/38077/{}'.format(filename), full_path, {'RequestPayer':'requester'})
    return full_path

if __name__ == '__main__':
  parameters_message = "parameters are: download"
  if len(sys.argv) == 1:
    print(parameters_message)
  elif sys.argv[1] == 'download':
  	naiper = NAIPDownloader()
  	naiper.download_naips()
  else:
    print(parameters_message)
