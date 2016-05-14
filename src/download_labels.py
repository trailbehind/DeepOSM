'''
Extract Ways from OSM PBF files
'''
import os
import time

import osmium as o
import requests
import shapely.wkb as wkblib

# http://docs.osmcode.org/pyosmium/latest/intro.html
# A global factory that creates WKB from a osmium geometry
wkbfab = o.geom.WKBFactory()

# set in Dockerfile as env variable
GEO_DATA_DIR = os.environ.get("GEO_DATA_DIR")


class WayMap():
    def __init__(self, extract_type='highway'):
        self.extracter = WayExtracter(extract_type)

    def extract_files(self, file_list):
        for path in file_list:
            self.run_extraction(path)

    def run_extraction(self, file_path):
        t0 = time.time()
        self.extracter.apply_file(file_path, locations=True)
        t1 = time.time()
        elapsed = "{0:.1f}".format(t1 - t0)
        print "EXTRACTED WAYS with locations from pbf file {}, took {}s".format(file_path, elapsed)


class WayExtracter(o.SimpleHandler):
    def __init__(self, extract_type='highway'):
        '''
            extract_type can so far be in: highway, tennis
        '''
        o.SimpleHandler.__init__(self)
        self.ways = []
        self.way_dict = {}
        self.types = []
        self.extract_type = extract_type

    def way(self, w):
        if self.extract_type == 'tennis':
            self.extract_if_tennis_court(w)
        else:
            self.extract_way_type(w)

    def extract_if_tennis_court(self, w):
        is_tennis = False
        for tag in w.tags:
            if tag.k == 'sport' and 'tennis' == tag.v:
                is_tennis = True

        if not is_tennis:
            return

        way_dict = {'uid': w.uid,
                    'ends_have_same_id': w.ends_have_same_id(),
                    'id': w.id,
                    'tags': []}

        for tag in w.tags:
            way_dict['tags'].append((tag.k, tag.v))

        self.add_linestring(w, way_dict)

    def extract_way_type(self, w):
        should_extract = False
        way_type = None
        for tag in w.tags:
            if tag.k == self.extract_type:
                way_type = tag.v
                should_extract = True

        if not should_extract:
            return

        if way_type not in self.types:
            self.types.append(way_type)

        way_dict = {'visible': w.visible,
                    'deleted': w.deleted,
                    'uid': w.uid,
                    'way_type': way_type,
                    'ends_have_same_id': w.ends_have_same_id(),
                    'id': w.id,
                    'tags': []}
        for tag in w.tags:
            way_dict['tags'].append((tag.k, tag.v))
        self.add_linestring(w, way_dict)

    def add_linestring(self, w, way_dict):
        try:
            wkb = wkbfab.create_linestring(w)
        except:
            # throws on single point ways
            return
        line = wkblib.loads(wkb, hex=True)
        reverse_points = []
        for point in list(line.coords):
            reverse_points.append([point[1], point[0]])
        way_dict['linestring'] = reverse_points
        self.ways.append(way_dict)


def download_and_extract(file_urls_to_download, extract_type='highway'):
    file_urls = file_urls_to_download
    file_paths = download_files(file_urls)
    w = WayMap(extract_type=extract_type)
    w.extract_files(file_paths)
    return w


def download_file(url):
    local_filename = url.split('/')[-1]
    full_local_filename = os.path.join(GEO_DATA_DIR, local_filename)
    r = requests.get(url, stream=True)
    with open(full_local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
    return full_local_filename


def download_files(url_list):
    paths = []
    print("DOWNLOADING {} PBFs...".format(len(url_list)))
    t0 = time.time()
    for url in url_list:
        local_filename = url.split('/')[-1]
        full_local_filename = os.path.join(GEO_DATA_DIR, local_filename)
        if not os.path.exists(full_local_filename):
            paths.append(download_file(url))
        else:
            paths.append(full_local_filename)
            print("PBF {} already downloaded".format(full_local_filename))
    if time.time() - t0 > 0.01:
        print("downloads took {0:.1f}s".format(time.time() - t0))
    return paths
