'''
Extract Ways in an OSM PBF file
NOT DONE: clip them to the bounds of tiles

Based on: 
  make nodecache: https://github.com/osmcode/pyosmium/blob/master/examples/create_nodecache.py
  use nodecache: https://github.com/osmcode/pyosmium/blob/master/examples/use_nodecache.py 
'''

import osmium as o
import sys

class WayMap():
    def __init__(self):
      self.ways = []

    def run_extraction(self, file_path):
      # cache locations for ways
      reader = o.io.Reader(file_path, o.osm.osm_entity_bits.NODE)
      node_cache = "sparse_file_array," + "data/tmp_node_cache"
      idx = o.index.create_map(node_cache)
      lh = o.NodeLocationsForWays(idx)
      o.apply(reader, lh)
      reader.close()
    
      # extract ways
      reader = o.io.Reader(file_path, o.osm.osm_entity_bits.WAY)
      idx = o.index.create_map(node_cache)
      lh = o.NodeLocationsForWays(idx)
      lh.ignore_errors()
      extracter = WayExtracter(idx)
      o.apply(reader, lh, extracter)
      reader.close()
      self.extracter = extracter

class WayExtracter(o.SimpleHandler):
    def __init__(self, idx):
        o.SimpleHandler.__init__(self)
        self.ways = []

    def way(self, w):
        x = 0
        for n in w.nodes:
          try:
            #print("{} {}, {}".format(x, n.lat, n.lon))
            self.ways.append(w)
          except:
            continue
            #print("no location for node {}".format(n))
          x += 1
          if x == 1:
            return

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python extract_ways.py <osmfile>")
        sys.exit(-1)

    w = WayMap()
    w.run_extraction(sys.argv[1])