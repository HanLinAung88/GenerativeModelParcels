import shapefile
import matplotlib.pyplot as plt
import numpy as np
import random
import fiona
import json
from pyproj import Proj, transform
#file_name = 'PARCELLES_GRAPHIQUES.shp'
file_name = 'PARCELLES_GRAPHIQUES' + '.shx'
sf = shapefile.Reader(file_name)


'''
Metadata information
'''
print(sf)
print('Shapetype: ', sf.shapeType)
print('Num of records: ', len(sf))
print('Fields :' ,sf.fields)
field_names = [field[0] for field in sf.fields]
'''
Shape file reading methods
'''
#first feature of shape file (including shape and record)
feature = sf.shapeRecord(0)
shape = sf.shape(0)
record = sf.record(0)
record_dict = record.as_dict()
#converts to GeoJson format
buffer = []
first = feature.shape.__geo_interface__  
first_shape = shape.__geo_interface__
atr = dict(zip(field_names, record))
buffer.append(dict(type="Feature", geometry=first_shape, properties=atr))
print(buffer)
assert(first == first_shape)
print('First feature or first shape: ', first)
print('First record dict: ', record_dict)

def draw_im(shape, shape_file='PARCELLES_GRAPHIQUES.shp', destination_proj='espg:4326'):
  my_dpi = 128
  original = Proj(fiona.open(shape_file).crs)
  destination = Proj('epsg:4326')
  fig = plt.figure(figsize=(1000/my_dpi, 1000/my_dpi), dpi=my_dpi, frameon=False)
  x_lon = np.zeros((len(shape.points),1))
  y_lat = np.zeros((len(shape.points),1))
  for ip in range(len(shape.points)):
    x_lon[ip] = shape.points[ip][0]
    y_lat[ip] = shape.points[ip][1]
    long, lat = transform (original , destination, shape.points[ip][0], shape.points[ip][1])
    x_lon[ip] = lat
    y_lat[ip] = long
  ax = plt.Axes(fig, [0., 0., 1., 1.])
  ax.set_axis_off()
  fig.add_axes(ax)
  plt.plot(x_lon,y_lat,'k')
  fig.savefig('example.png', dpi=my_dpi)


def draw_im_geo_json(geo_json_file='pyshp-1.json', destination_proj='espg:4326'):
  with open(geo_json_file) as f:
    geo_dict = json.load(f)
  my_dpi = 128
  fig = plt.figure(figsize=(1000/my_dpi, 1000/my_dpi), dpi=my_dpi, frameon=False)
  for feature in geo_dict['features']:
    coordinates = np.array(feature['geometry']['coordinates'][0])
    x_lon = coordinates[:, 0]
    y_lat = coordinates[:, 1]    
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.plot(x_lon,y_lat,'k')
    fig.savefig('geo_json.png', dpi=my_dpi)

#draw_im(shape)
draw_im_geo_json()
