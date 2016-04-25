'''
    a class to download NAIP imagery from 
    the aws-naip RequesterPays bucket
'''

import boto3
from random import shuffle
import sys, os, subprocess

GEO_DATA_DIR = os.environ.get("GEO_DATA_DIR") # set in Dockerfile as env variable
NAIP_DATA_DIR = os.path.join(GEO_DATA_DIR, "naip")

class NAIPDownloader:

  def __init__(self):
  	self.make_directory(NAIP_DATA_DIR, full_path=True)

  def make_directory(self, new_dir, full_path=False):
    '''
       make a new directory tree if it doesn't already exist
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
    '''
        download 4 randomish NAIPs from aws-naip bucket
    '''

    # configure s3cmd with AWS credentials
    file_path = '/' + os.environ.get("HOME")+'/.s3cfg'
    f = open(file_path,'r')
    filedata = f.read()
    f.close()
    newdata = filedata.replace("AWS_ACCESS_KEY",os.environ.get("AWS_ACCESS_KEY_ID"))
    newdata = newdata.replace("AWS_SECRET_KEY",os.environ.get("AWS_SECRET_ACCESS_KEY"))
    f = open(file_path,'w')
    f.write(newdata)
    f.close()

    # list the contents of the bucket directory
    bashCommand = "s3cmd ls --recursive --skip-existing s3://aws-naip/md/2013/1m/rgbir/ --requester-pays"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output = process.communicate()[0]
    
    # make a list of 4 random NAIPs from Maryland 2013,
    # from area '38077' which I assume is a grid number for now
    state = 'md'
    year = '2013'
    resolution = '1m'
    spectrum = 'rgbir' 
    grid = '38077'
    naip_filenames = []
    for line in output.split('\n'):
      try:
        fn = line.split('s3://aws-naip/{}/{}/{}/{}/{}/'.format(state,year,resolution,spectrum,grid))[1]
        naip_filenames.append(fn)
      except:
        pass 
        # print "WARNING: no problem, skipping a line in the ls response from AWS"
    shuffle(naip_filenames)

    # download the NAIPs
    s3_client = boto3.client('s3')
    naip_local_paths = []
    for filename in naip_filenames[0:4]:
      full_path = os.path.join(NAIP_DATA_DIR, filename)
      if os.path.exists(full_path):
        print("NAIP {} already downloaded".format(full_path))
      else:
        s3_url = '{}/{}/{}/{}/{}/{}'.format(state, year, resolution, spectrum, grid, filename)
        s3_client.download_file('aws-naip', s3_url, full_path, {'RequestPayer':'requester'})
      naip_local_paths.append(full_path)

    return naip_local_paths

if __name__ == '__main__':
  parameters_message = "parameters are: download"
  if len(sys.argv) == 1:
    print(parameters_message)
  elif sys.argv[1] == 'download':
  	naiper = NAIPDownloader()
  	naiper.download_naips()
  else:
    print(parameters_message)
