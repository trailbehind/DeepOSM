'''
Extract Ways in an OSM PBF file, and map the way-nodeid-lists to coordinates
so they can be searched and clipped 

NOT DONE: clip them to the bounds of tiles
'''

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

    print('{} ways in this extract'.format(len(h.ways)))
    print('way 0 is {}'.format(h.ways[0]))

'''

from imposm.parser import OSMParser

class WayExtracter(object):
  way_list = []
  node_location_map = {}
  relation_map = {}

  def nodes(self, nodes):
    for node in nodes:
      node_id, info, coord_tuple = node
      self.node_location_map[node_id] = coord_tuple

  def ways(self, ways):
    for way in ways:
      self.way_list.append(way)

  def relations(self, relation):
    print(relation)
    for relation in relations:
      print(relation)
      relation_id, info, ref_list = relation
      self.relation_map[relation_id] = ref_list
    print("relations {}".format(len( self.relation_map)))

  def fuse_coordinates_to_ways(self):
    #self.updated_ways = []
    for way in self.way_list:
      way_id, info, relation_id_list = way
      linestring = []
      for relation_id in relation_id_list:
        try:
          print(self.relation_map[relation_id])
        except:
          print("cant find relation")  
          try:
            print(self.node_location_map[relation_id])
          except:
            print("cant find node either")  
        #linestring.append(self.node_location_map[node_id])
      #self.updated_ways.append(linestring)

extracter = WayExtracter()
p = OSMParser(concurrency=4, 
              ways_callback=extracter.ways, 
              nodes_callback=extracter.nodes,
              relations_callback=extracter.relations)
p.parse('data/california-latest.osm.pbf')

extracter.fuse_coordinates_to_ways()
print("mapped {} linestrings".format(len(extracter.updated_ways)))
print("linestring 0 is {}".format(extracter.updated_ways[0]))
# done
print("extracted {} ways".format(len(extracter.way_list)))
print("way 0 us {}".format(extracter.way_list[0]))
'''