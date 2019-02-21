import xml.etree.ElementTree as ET
from pathlib import Path
import os,sys
import numpy as np
import pandas as pd
import geopy
import geopy.distance

# Set variables
path = Path('/flashblade/lars_data/2018_Cyclomedia_panoramas/project_folder')
annotpath = path / 'YOLO/borden_train_balanced/valid_img/annot'
metapath = path / 'YOLO/borden_train_balanced/metadata'
outputpath = path / 'output'

# Make table with all the real heights of the borden in mm
heights = [1000,1180,1000,1000,1000,1300,900,900,900,800,900] # Last 6 are estimates, as there are multiple versions of the bord
classes = ['A01_80','G01','A08_80','A01_100','A01_100s','A01_100o','J01','J27','B06','C02','L05']
#['A01-100o','A08-80','A01-80','G01','J01','C02','J27','A01-100s','A01-100','B06','L05']
heights_table = pd.DataFrame({'heights (mm)':heights},index=classes)

# List all the files in the directory
filelist = os.listdir(str(annotpath))

# create the empty arrays that will be the columns
classes = []
lats = []
lons = []
xmins = []
xmaxs = []
ymins = []
ymaxs = []
yaws = []

# Loop over xml files in annotpath
for ann in sorted(filelist):
    if ann.endswith('xml'):
#        print(ann)
        lat = float(ann.split('_')[-1].replace('.xml',''))
        lon = float(ann.split('_')[-2])
        img = {'object':[]}
        
        # Open metafile to obtain the yaw
        metafile = ann.split('_')[2] + '_' + ann.split('_')[3] + '.txt'
#        print(str(metapath / metafile))
        try:
            with open(str(metapath / metafile)) as file:
                for line in file:
                    if line.startswith('yaw='):
                        yaw=float(line.split('"')[-2])
    
            tree = ET.parse(os.path.join(str(annotpath),ann))
            for elem in tree.iter():
                if 'object' in elem.tag or 'part' in elem.tag:
                    obj = {}
    
                    for attr in list(elem):
                        if 'name' in attr.tag:
                            obj['name'] = attr.text
                            bord = obj['name']
    
                        if 'bndbox' in attr.tag:
                            for dim in list(attr):
                                if 'xmin' in dim.tag:
                                    obj['xmin'] = int(round(float(dim.text)))
                                if 'ymin' in dim.tag:
                                    obj['ymin'] = int(round(float(dim.text)))
                                if 'xmax' in dim.tag:
                                    obj['xmax'] = int(round(float(dim.text)))
                                if 'ymax' in dim.tag:
                                    obj['ymax'] = int(round(float(dim.text)))
                                    
                            classes = np.append(classes,obj['name'].replace('-','_'))
                            lats = np.append(lats,lat)
                            lons = np.append(lons,lon)
                            xmins = np.append(xmins,obj['xmin'])
                            xmaxs = np.append(xmaxs,obj['xmax'])
                            ymins = np.append(ymins,obj['ymin'])
                            ymaxs = np.append(ymaxs,obj['ymax'])
                            yaws = np.append(yaws,yaw)
        except:
            print(metafile)
# Create dataframe with all the info for each bord                        
annot_borden_db = pd.DataFrame({'class':classes,
                               'latitude':lats,
                               'longitude':lons,
                               'xmin':xmins,
                               'xmax':xmaxs,
                               'ymin':ymins,
                               'ymax':ymaxs,
                               'yaw':yaws})       

# Make "borden-id"
annot_borden_db['borden_id']= np.arange(len(annot_borden_db))

# Update location function
def update_location(sign_class,lat,lon,xmin,xmax,ymin,ymax,yaw,heights_table):
    image_height = 512
    f_over_sensor = 0.87
    x_mid = (xmax+xmin)/2
    angle = yaw-45+(90/1024)*x_mid
    # lookup real height
    real_height = heights_table.loc[sign_class,'heights (mm)']
    # Calculate distance from bbox
    distance_m = ((real_height*image_height*f_over_sensor)/(ymax-ymin))/1000
    
    # Give the sign the new geolocation
    start = geopy.Point(lat,lon)
    d = geopy.distance.distance(meters = distance_m)
    no_d = geopy.distance.distance(meters = 0)
    new_location = d.destination(point=start, bearing=angle)
    start_coords = no_d.destination(point=start, bearing=0)
    new_lat = new_location.latitude
    new_lon = new_location.longitude
    return new_lat,new_lon



# loop over boxes and update the geolocation
for index in annot_borden_db.index:
    annot_borden_db.loc[index,'new_lat'],annot_borden_db.loc[index,'new_lon']= update_location(annot_borden_db.loc[index,'class'],
                          annot_borden_db.loc[index,'latitude'],
                          annot_borden_db.loc[index,'longitude'],
                          annot_borden_db.loc[index,'xmin'],
                          annot_borden_db.loc[index,'xmax'],
                          annot_borden_db.loc[index,'ymin'],
                          annot_borden_db.loc[index,'ymax'],
                          annot_borden_db.loc[index,'yaw'],
                          heights_table)

# Check if bord with same class is within 10m of bord above them in the list, if yes: give them the same "borden-id"
for index in annot_borden_db.index:
    indices_above=np.arange(index-1,-1,-1)
    for index_above in indices_above:
        if annot_borden_db.loc[index,'class']==annot_borden_db.loc[index_above,'class']:
            coord1 = (annot_borden_db.loc[index,'new_lat'],annot_borden_db.loc[index,'new_lon'])
            coord2 = (annot_borden_db.loc[index_above,'new_lat'],annot_borden_db.loc[index_above,'new_lon'])
            distance = geopy.distance.distance(coord1, coord2).m
            if distance < 10:
                annot_borden_db.loc[index,'borden_id']=annot_borden_db.loc[index_above,'borden_id']
print(annot_borden_db)
# Group the borden with the same "borden_id" and write as annot_borden_db.csv
final_annot_borden_db=(annot_borden_db.groupby('borden_id').agg({'class':'first',
                                               'new_lat':'mean',
                                               'new_lon':'mean'}))

print(final_annot_borden_db)
final_annot_borden_db.to_csv(outputpath / 'annot_borden_valid_db.csv',header=['longitude','class','latitude'])
