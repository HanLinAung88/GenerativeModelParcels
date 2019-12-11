import shapefile
import matplotlib.pyplot as plt
import numpy as np
import random

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

'''
Shape file drawing methods
'''
#https://chrishavlin.com/2016/11/16/shapefiles-tutorial/
#Single shape

x_lon = np.zeros((len(shape.points),1))
y_lat = np.zeros((len(shape.points),1))
for ip in range(len(shape.points)):
  x_lon[ip] = shape.points[ip][0]
  y_lat[ip] = shape.points[ip][1]
plt.plot(x_lon,y_lat,'k')

plt.savefig('First.png')
#plt.show()
#Multiple shapes (100)

index = 0
for shape in sf.iterShapes():
  shape.__geo_interface__
  if random.random() < 0.1:
    npoints=len(shape.points) # total points
    nparts = len(shape.parts) # total parts
    if index > 10000:
      break
    if nparts == 1:
      x_lon = np.zeros((len(shape.points),1))
      y_lat = np.zeros((len(shape.points),1))
      for ip in range(len(shape.points)):
        x_lon[ip] = shape.points[ip][0]
        y_lat[ip] = shape.points[ip][1]
      plt.plot(x_lon,y_lat)

    else: # loop over parts of each shape, plot separately
      for ip in range(nparts): # loop over parts, plot separately
        i0=shape.parts[ip]
        if ip < nparts-1:
          i1 = shape.parts[ip+1]-1
        else:
          i1 = npoints
      seg=shape.points[i0:i1+1]
      x_lon = np.zeros((len(seg),1))
      y_lat = np.zeros((len(seg),1))
      for ip in range(len(seg)):
        x_lon[ip] = seg[ip][0]
        y_lat[ip] = seg[ip][1]
      plt.plot(x_lon, y_lat)
    index += 1

print('Finished')
plt.show()
plt.savefig('All.png')



'''
Dealing with massive shape files
'''
'''
for shape in sf.iterShapes():
  pass
for rec in sf.iterRecords():
  pass
for shapeRec in sf.iterShapeRecords():
  pass
for shapeRec in sf: # same as iterShapeRecords()
  pass
'''
