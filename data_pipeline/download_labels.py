'''
Extract Ways from OSM PBF files
'''

import osmium as o
import json, os, requests, sys, time
import shapely.wkb as wkblib

# http://docs.osmcode.org/pyosmium/latest/intro.html
# A global factory that creates WKB from a osmium geometry
wkbfab = o.geom.WKBFactory()

GEO_DATA_DIR = os.environ.get("GEO_DATA_DIR") # set in Dockerfile as env variable

# enough to cover a NAIP around DC
DEFAULT_FILE_URLS = ['http://download.geofabrik.de/north-america/us/maryland-latest.osm.pbf',
                     'http://download.geofabrik.de/north-america/us/district-of-columbia-latest.osm.pbf']

class WayMap():
    def __init__(self):
      pass

    def extract_files(self, file_list):
      extracter = WayExtracter()
      self.extracter = extracter
      for path in file_list:
        print "EXTRACTING PBF {}".format(path)
        self.run_extraction(path)

    def run_extraction(self, file_path):
      # extract ways

      cache_path = file_path + '.cache'

      if os.path.exists(cache_path):
        t0 = time.time()
        with open(cache_path, 'w') as outfile:
          self.extracter.ways = json.load(outfile)
        t1 = time.time()      
        elapsed = "{0:.2f}".format(t1-t0)
        print "USING CACHED WAYS from pbf file {}, fetched from disk in {}".format(file_path, elapsed)
        return

      t0 = time.time()
      self.extracter.apply_file(file_path, locations=True)
      t1 = time.time()      
      elapsed = "{0:.2f}".format(t1-t0)
      print "EXTRACTED WAYS with locations from pbf file {}, took {}".format(file_path, elapsed)
      with open(cache_path, 'w') as outfile:
        json.dump(self.extracter.ways, outfile)

class WayExtracter(o.SimpleHandler):
    def __init__(self):
        o.SimpleHandler.__init__(self)
        self.ways = []

    def way(self, w):
        is_highway = False
        is_big = False
        name = ''
        highway_type = None
        self.types = []

        for tag in w.tags:
          if tag.k == 'name':
            name = tag.v
          #  and tag.v in ['primary', 'secondary', 'tertiary', 'trunk']
          if tag.k == 'highway':
            highway_type = tag.v
            is_highway = True
          #try:
          #  if tag.k == 'lanes' and int(tag.v[len(tag.v)-1]) >= 2:
          #    is_big = True
          #  #    #for t in w.tags:
          #  #    #  print "tag {} {}".format(t.k, t.v)
          #except:
          #  print("exception, weird lanes designation {}".format(tag.v))

        #  or not is_big
        if not is_highway:
          return
        
        if not highway_type in self.types:
          self.types.append(highway_type)

        way_dict = {'visible': w.visible,
                    'deleted': w.deleted,
                    'uid': w.uid,
                    'highway_type': highway_type,
                    'ends_have_same_id': w.ends_have_same_id(),
                    'id': w.id,
                    'tags':[]}
        for tag in w.tags:
          way_dict['tags'].append((tag.k, tag.v))

        try:
          wkb = wkbfab.create_linestring(w)
        except:
          # throws on single point ways
          return
        line = wkblib.loads(wkb, hex=True)
        reverse_points = []
        for point in list(line.coords):
          reverse_points.append([point[1],point[0]])
        way_dict['linestring'] = reverse_points
        self.ways.append(way_dict)

def download_and_extract():
    file_urls = DEFAULT_FILE_URLS
    file_paths = download_files(file_urls)               
    w = WayMap()
    w.extract_files(file_paths)
    return w

def download_file(url):
    local_filename = url.split('/')[-1]
    full_local_filename = os.path.join(GEO_DATA_DIR, local_filename)
    r = requests.get(url, stream=True)
    with open(full_local_filename, 'wb') as f:
      for chunk in r.iter_content(chunk_size=1024):
        if chunk: # filter out keep-alive new chunks
          f.write(chunk)
    return full_local_filename

def download_files(url_list):
  paths = []
  for url in url_list:
    local_filename = url.split('/')[-1]
    full_local_filename = os.path.join(GEO_DATA_DIR, local_filename)
    if not os.path.exists(full_local_filename):
      paths.append(download_file(url))
    else:
      paths.append(full_local_filename)
      print("PBF {} already downloaded".format(full_local_filename))
  return paths

