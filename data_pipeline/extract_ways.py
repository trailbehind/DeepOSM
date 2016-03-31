'''
Extract Ways in an OSM PBF file
NOT DONE: clip them to the bounds of tiles

Based on: 
  make nodecache: https://github.com/osmcode/pyosmium/blob/master/examples/create_nodecache.py
  use nodecache: https://github.com/osmcode/pyosmium/blob/master/examples/use_nodecache.py 
'''

import osmium as o
import sys

class WayExtracter(o.SimpleHandler):
    def __init__(self, idx):
        o.SimpleHandler.__init__(self)
        self.ways = []

    def way(self, w):
        self.ways.append(w)
        for n in w.nodes:
          try:
            print("{}, {}".format(n.lat, n.lon))
          except:
            print("no location for node {}".format(n))
          
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python extract_ways.py <osmfile>")
        sys.exit(-1)

    # cache locations for ways
    reader = o.io.Reader(sys.argv[1], o.osm.osm_entity_bits.NODE)
    node_cache = "sparse_file_array," + "data/tmp_node_cache"
    idx = o.index.create_map(node_cache)
    lh = o.NodeLocationsForWays(idx)
    o.apply(reader, lh)
    reader.close()
    
    # extract ways
    reader = o.io.Reader(sys.argv[1], o.osm.osm_entity_bits.WAY)
    idx = o.index.create_map(node_cache)
    lh = o.NodeLocationsForWays(idx)
    lh.ignore_errors()
    o.apply(reader, lh, WayExtracter(idx))
    reader.close()