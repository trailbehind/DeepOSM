"""A class to download NAIP imagery from the s3://aws-naip RequesterPays bucket."""

import os
import subprocess
import sys
import time
from random import shuffle

import boto3

# set in Dockerfile as env variable
GEO_DATA_DIR = os.environ.get("GEO_DATA_DIR")
NAIP_DATA_DIR = os.path.join(GEO_DATA_DIR, "naip")


class NAIPDownloader:
    """Downloads NAIP images from S3, by state/year."""

    def __init__(self, number_of_naips, should_randomize, state, year):
        """Download some arbitrary NAIP images from the aws-naip S3 bucket."""
        self.number_of_naips = number_of_naips
        self.should_randomize = should_randomize

        self.state = state
        self.year = year
        self.resolution = '1m'
        self.spectrum = 'rgbir'
        self.bucket_url = 's3://aws-naip/'

        self.url_base = '{}{}/{}/{}/{}/'.format(self.bucket_url, self.state, self.year,
                                                self.resolution, self.spectrum)

        self.make_directory(NAIP_DATA_DIR, full_path=True)

    def make_directory(self, new_dir, full_path=False):
        """Make a new directory tree if it doesn't already exist."""
        if full_path:
            path = ''
            for token in new_dir.split('/'):
                path += token + '/'
                try:
                    os.mkdir(path)
                except:
                    pass
            return path

        try:
            os.mkdir(new_dir)
        except:
            pass
        return new_dir

    def download_naips(self):
        """Download self.number_of_naips of the naips for a given state."""
        self.configure_s3cmd()
        naip_filenames = self.list_naips()
        if self.should_randomize:
            shuffle(naip_filenames)
        naip_local_paths = self.download_from_s3(naip_filenames)
        return naip_local_paths

    def configure_s3cmd(self):
        """Configure s3cmd with AWS credentials."""
        file_path = os.environ.get("HOME") + '/.s3cfg'
        f = open(file_path, 'r')
        filedata = f.read()
        f.close()
        newdata = filedata.replace("AWS_ACCESS_KEY", os.environ.get("AWS_ACCESS_KEY_ID"))
        newdata = newdata.replace("AWS_SECRET_KEY", os.environ.get("AWS_SECRET_ACCESS_KEY"))
        f = open(file_path, 'w')
        f.write(newdata)
        f.close()

    def list_naips(self):
        """Make a list of NAIPs based on the init parameters for the class."""
        # list the contents of the bucket directory
        bash_command = "s3cmd ls --recursive --skip-existing {} --requester-pays".format(
            self.url_base)
        process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
        output = process.communicate()[0]
        naip_filenames = []
        for line in output.split('\n'):
            parts = line.split(self.url_base)
            # there may be subdirectories for each state, where directories need to be made
            if len(parts) == 2:
                naip_path = parts[1]
                naip_filenames.append(naip_path)
                naip_subpath = os.path.join(NAIP_DATA_DIR, naip_path.split('/')[0])
                if not os.path.exists(naip_subpath):
                    os.mkdir(naip_subpath)
            else:
                pass
                # skip non filename lines from response

        return naip_filenames

    def download_from_s3(self, naip_filenames):
        """Download the NAIPs and return a list of the file paths."""
        s3_client = boto3.client('s3')
        naip_local_paths = []
        max_range = self.number_of_naips
        if max_range == -1:
            max_range = len(naip_filenames)
        t0 = time.time()
        has_printed = False
        for filename in naip_filenames[0:max_range]:
            # for filename in ['m_3807708_ne_18_1_20130924.tif']:
            full_path = os.path.join(NAIP_DATA_DIR, filename)
            if os.path.exists(full_path):
                print("NAIP {} already downloaded".format(full_path))
            else:
                if not has_printed:
                    print("DOWNLOADING {} NAIPs...".format(max_range))
                    has_printed = True
                url_without_prefix = self.url_base.split(self.bucket_url)[1]
                s3_url = '{}{}'.format(url_without_prefix, filename)
                s3_client.download_file('aws-naip', s3_url, full_path, {'RequestPayer': 'requester'
                                                                        })
            naip_local_paths.append(full_path)
        if time.time() - t0 > 0.01:
            print("downloads took {0:.1f}s".format(time.time() - t0))
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
