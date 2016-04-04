'''
   common data structures for geo/raster data
'''

class BoundingBox:
  def __init__(self):
    self.northeast = Coordinate()
    self.southwest = Coordinate()


class Coordinate:
  def __init__(self, lat=-999, lon=-999):
    self.lat = lat
    self.lon = lon

  def __str__(self):
    return "{} {}".format(self.lat, self.lon)


class MercatorTile:
  def __init__(self, x=-1, y=-1, z=-1):
    self.x = x
    self.y = y 
    self.z = z

  def __str__(self):
    return "{} {} {}".format(self.x, self.y, self.z)


class Pixel:
  def __init__(self, x=0, y=0):
    self.x = x
    self.y = y

  def __str__(self):
    return "{} {}".format(self.x, self.y)

