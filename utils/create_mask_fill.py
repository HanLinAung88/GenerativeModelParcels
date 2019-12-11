import json
import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt
from PIL import Image


def read_csv(csv_file='data/testFR_450mbbox.csv'):
  grid = dict()
  keys = ['maxlat', 'maxlon', 'minlat', 'minlon']
  with open(csv_file) as f:
    readCSV = csv.reader(f)
    for index, row in enumerate(readCSV, -1):
      if index == -1:
        continue
      if index not in grid:
        grid[index] = dict()
        for key in keys:
          grid[index][key] = 0
      grid[index]['Parcel_id'] = float(row[2])
      grid[index]['maxlat'] = float(row[3])
      grid[index]['maxlon'] = float(row[4])
      grid[index]['minlat'] = float(row[5])
      grid[index]['minlon'] = float(row[6])
  return grid

def point_is_in_bounds(point, w, h):
  if point[0] >= 0 and points[0] > w and point[1] >= 0 and point[1] <= h:
    return True
  return False

def scale_coords(shape_size, geom, grid, index, size_m = 450):
  w, h = shape_size
  min_lat, min_lon, max_lat, max_lon = grid[index]['minlat'], grid[index]['minlon'], grid[index]['maxlat'], grid[index]['maxlon']
  x = geom[:,0]
  y = geom[:,1]
  w = 1500
  h = 1500
  scale_lon = w/(max_lon - min_lon)
  scale_lat = h/(max_lat-min_lat)
  scaled_x = (x - min_lon) * scale_lon # lon-> x, lat->y
  scaled_y = h - ((y - min_lat) * scale_lat)
  if any(val > w for val in scaled_x) or any(val > h for val in scaled_y) or any(val < 0 for val in scaled_x) or any (val < 0 for val in scaled_y):
     return False, np.concatenate([scaled_x[:,None], scaled_y[:,None]],axis=1)
  return True, np.concatenate([scaled_x[:,None], scaled_y[:,None]],axis=1)


base_dir = '/atlas/u/hanlaung/ParcelDelineation/data/'
csv_file = base_dir + 'testFR_450mbbox.csv'
json_file = base_dir + 'json_files/' + 'pyshp-all-2000fit.json'
with open(json_file) as f:
  shp_dict = json.load(f)

shape_size = (1500, 1500)
grid = read_csv(csv_file)

for index in range(len(grid.keys())):
  parcel_id = grid[index]['Parcel_id']
  polys = []
  for sh_index, sh in enumerate(shp_dict['features']):
    for coord_idx in range(len(sh['geometry']['coordinates'])):
      geom = np.array(sh['geometry']['coordinates'][coord_idx])
      is_in_bounds, geom_fixed = scale_coords(shape_size, geom, grid, index)
      pts = geom_fixed.astype(int)
      polys.append(pts)

  #Creates the binary mask
  mask = np.zeros(shape_size)
  cv2.fillPoly(mask, polys, color=255)
  cv2.polylines(mask, polys, True, color=255,thickness=10)
#  mask[mask == 255] = 1
#  mask[mask != 1] = 0

  #Saves the binary mask file
  cv2.imwrite(base_dir + 'train_masks_filled/image_binary_' + str(int(parcel_id)) + '.png', np.array(mask))

  #Saves the overlay file
  '''
  im_name = base_dir + 'train_filled/DG_' + str(int(parcel_id)) +'.jpeg'
  print(im_name)
  orig_image = cv2.imread(im_name)
  cv2.polylines(orig_image, polys, True, color=(255,255,255),thickness=10)
  cv2.imwrite(base_dir + 'overlay/image_overlay_' + str(int(parcel_id)) + '.jpeg', orig_image)
  '''
  # To show the images
  # plt.imshow(orig_image)
  # plt.show()
  # plt.close()

