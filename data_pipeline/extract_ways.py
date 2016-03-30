'''
Extract Ways in an OSM PBF file
NOT DONE: clip them to the bounds of tiles
Based on: https://github.com/osmcode/pyosmium/blob/master/examples/road_length.py 
'''

import osmium as o
import sys

class WayExtracter(o.SimpleHandler):
    def __init__(self):
        o.SimpleHandler.__init__(self)
        self.ways = []

    def way(self, w):
        self.ways.append(w)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python extract_ways.py <osmfile>")
        sys.exit(-1)

    h = WayExtracter()
    # set 'locations' to true so we get all the nodes, 
    # to calculate which Ways cross which tiles
    h.apply_file(sys.argv[1], locations=True)

    print('{} ways in this extract'.format(h.ways))
