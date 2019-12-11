import json
import random

def explode(coords):
  for e in coords:
    if isinstance(e, (float, int)):
      yield coords
      break
    else:
      for f in explode(e):
        yield f

def bbox(f):
  x, y = zip(*list(explode(f['geometry']['coordinates'])))
  return min(x), min(y), max(x), max(y)

file_name = sys.argv[1] #e.g. Name of json file for geo_json format:  'pyshp-100.json'
centroid_file_name = sys.argv[2] # e.g. Name of output file with centroids: 'centroid-100.txt'

with open(file_name) as f:
  geo_dict = json.load(f)

centroids = []

for feature in geo_dict['features']:
  coords = bbox(feature)
  coordinates = feature['geometry']['coordinates'][0]
  centroid = [(coords[0] + coords[2])/2, (coords[1] + coords[3])/2]
  centroids.append(centroid)


with open('centroid-100.txt', 'w') as f:
  f.write('Latitude, Longitude \n')
  for centroid in centroids:
    f.write(str(centroid[1]) + ',' + str(centroid[0]) + "\n")
