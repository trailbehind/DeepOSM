'''
Extract Ways in an OSM PBF file
'''

import osmium as o
import sys, os, requests
import shapely.wkb as wkblib

# http://docs.osmcode.org/pyosmium/latest/intro.html
# A global factory that creates WKB from a osmium geometry
wkbfab = o.geom.WKBFactory()

class WayMap():
    def __init__(self):
      pass

    def run_extraction(self, file_path):
      # extract ways
      extracter = WayExtracter()
      extracter.apply_file(file_path, locations=True)
      self.extracter = extracter
      return
      for key in self.extracter.way_dict:
        combined_line = {'id':key, 'linestring':[]}
        for way_dict in self.extracter.way_dict[key]:
          for point in way_dict['linestring']:
            combined_line['linestring'].append(point)
        self.extracter.ways.append(combined_line)

      print "EXTRACTED WAYS with locations from pbf file {}".format(file_path)

class WayExtracter(o.SimpleHandler):
    def __init__(self):
        o.SimpleHandler.__init__(self)
        self.ways = []
        self.way_dict = {}

    '''
    def relation(self, r):
          relation_dict = {'id': r.id}
          for tag in r.tags:
            print "({} {})".format(tag.k, tag.v)        
          print '\n'
    '''

    def way(self, w): 
        is_highway = False
        name = ''

        for tag in w.tags:
          if tag.k == 'name':
            name = tag.v
          if tag.k == 'highway':
            is_highway = True
        if not is_highway:
          return

        way_dict = {'visible': w.visible, 
                    'deleted': w.deleted,
                    'uid': w.uid, 
                    'ends_have_same_id': w.ends_have_same_id(), 
                    'id': w.id, 
                    'tags':[]}
        for tag in w.tags:
          way_dict['tags'].append((tag.k, tag.v))
        
        wkb = wkbfab.create_linestring(w)
        line = wkblib.loads(wkb, hex=True)
        reverse_points = []
        for point in list(line.coords):
          reverse_points.append([point[1],point[0]])
        way_dict['linestring'] = reverse_points
        self.ways.append(way_dict)
   
def download_file(url):
    local_filename = url.split('/')[-1]
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
      for chunk in r.iter_content(chunk_size=1024): 
        if chunk: # filter out keep-alive new chunks
          f.write(chunk)
    return local_filename

if __name__ == '__main__':
    if len(sys.argv) == 1:
      file_path = download_file('http://download.geofabrik.de/north-america/us/district-of-columbia-latest.osm.pbf')
    elif len(sys.argv) != 2:
      print("Usage: python extract_ways.py <osmfile>")
      sys.exit(-1)
    else:
      file_path = sys.argv[1]

    w = WayMap()
    w.run_extraction(file_path)