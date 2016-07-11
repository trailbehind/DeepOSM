"""Extract Ways from OSM PBF files."""
import os
import time

import osmium as o
import requests
import shapely.wkb as wkblib
from src.config import RAW_LABEL_DATA_DIR

# http://docs.osmcode.org/pyosmium/latest/intro.html
# A global factory that creates WKB from a osmium geometry
wkbfab = o.geom.WKBFactory()


class WayMap():
    """Extract ways from OpenStreetMap PBF extracts."""

    def __init__(self, extract_type='highway'):
        """The extract_type can be highway, footway, cycleway, or tennis."""
        self.extracter = WayExtracter(extract_type)

    def extract_files(self, file_list):
        """Extract ways from each PBF in file_list."""
        for path in file_list:
            self.run_extraction(path)

    def run_extraction(self, file_path):
        """Extract ways from a PBF file at file_path."""
        t0 = time.time()
        self.extracter.apply_file(file_path, locations=True)
        t1 = time.time()
        elapsed = "{0:.1f}".format(t1 - t0)
        print "EXTRACTED WAYS with locations from pbf file {}, took {}s".format(file_path, elapsed)


class WayExtracter(o.SimpleHandler):
    """Subclass of osmium SimpleHandler to extract ways from OpenStreetMap PBF files."""

    def __init__(self, extract_type='highway'):
        """Extract ways from OpenStreetMap PBF files."""
        o.SimpleHandler.__init__(self)
        self.ways = []
        self.way_dict = {}
        self.types = []
        self.extract_type = extract_type

    def way(self, w):
        """Fire this callback when osmium parses a way in the PBF file."""
        if self.extract_type == 'tennis':
            self.extract_if_tennis_court(w)
        else:
            self.extract_way_type(w)

    def extract_if_tennis_court(self, w):
        """Extract the way of it has a 'sport' tag with value 'tennis'."""
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
        """Extract the way (w) if its type matches extract_type (highway, footway, or cycleway)."""
        should_extract = False
        way_type = None
        for tag in w.tags:
            if tag.k == self.extract_type:
                way_type = tag.v
                should_extract = True
            # for roads analysis, don't extract ways that don't allow vehicle access
            if self.extract_type == 'highway' and tag.k == 'motor_vehicle' and tag.v == 'no':
                return

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
        """Append the way_dict, with coords normalized to (lat,lon) instead of (lon,lat) pairs."""
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
    """Download PBFs file_urls_to_download, and extract ways that match extract_type."""
    file_paths = download_files(file_urls_to_download)
    w = WayMap(extract_type=extract_type)
    w.extract_files(file_paths)
    return w


def download_file(url):
    """Download a large file in chunks and return its local path."""
    local_filename = url.split('/')[-1]
    full_local_filename = os.path.join(RAW_LABEL_DATA_DIR, local_filename)
    r = requests.get(url, stream=True)
    with open(full_local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
    return full_local_filename


def download_files(url_list):
    """Download the PBF files in url_list, and return a list of local paths."""
    paths = []
    print("DOWNLOADING {} PBFs...".format(len(url_list)))
    t0 = time.time()
    for url in url_list:
        local_filename = url.split('/')[-1]
        full_local_filename = os.path.join(RAW_LABEL_DATA_DIR, local_filename)
        if not os.path.exists(full_local_filename):
            paths.append(download_file(url))
        else:
            paths.append(full_local_filename)
            print("PBF {} already downloaded".format(full_local_filename))
    if time.time() - t0 > 0.01:
        print("downloads took {0:.1f}s".format(time.time() - t0))
    return paths
