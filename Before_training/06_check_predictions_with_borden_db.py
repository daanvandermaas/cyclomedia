import os,sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopy
import geopy.distance

# Variable list:
path = Path('/flashblade/lars_data/2018_Cyclomedia_panoramas/project_folder')
predictionpath = path / 'YOLO/stuk_amsterdam/output'
metapath = path / 'YOLO/stuk_amsterdam'
annotpath = path / 'output'

# Make table with all the real heights of the borden in mm
heights = [1000,1180,1000,1000,1000,1300,900,900,900,800,900] # Last 6 are estimates, as there are multiple versions of the bord
classes = ['A01_80','G01','A08_80','A01_100','A01_100s','A01_100o','J01','J27','B06','C02','L05']
heights_table = pd.DataFrame({'heights (mm)':heights},index=classes)

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

predic_borden = pd.read_csv(str(predictionpath / 'output.txt'),names=['Cyclo_id','xmin','xmax','ymin','ymax','Class','confidence'])
predic_borden['Class'] = predic_borden.Class.str.replace('-','_')
#predic_borden = predic_borden[predic_borden['Cyclo_id']=='5D61PPAC_R']

for index in predic_borden.index:
    metafile = predic_borden.loc[index,'Cyclo_id'] + '.txt'
    try:
        with open(str(metapath / metafile)) as file:
            for line in file:
                if line.startswith('recording-location-lat='):
                    cyclo_lat = line.split('"')[-2]
                    predic_borden.loc[index,'Cyclo_lat'] = float(cyclo_lat)
                if line.startswith('recording-location-lon='):
                    cyclo_lon = line.split('"')[-2]
                    predic_borden.loc[index,'Cyclo_lon'] = float(cyclo_lon)
                if line.startswith('yaw='):
                    cyclo_yaw = line.split('"')[-2]
                    predic_borden.loc[index,'Cyclo_yaw'] = float(cyclo_yaw)
        predic_borden.loc[index,'bord_lat'],predic_borden.loc[index,'bord_lon'] = update_location(predic_borden.loc[index,'Class'],
                                                                                                 predic_borden.loc[index,'Cyclo_lat'],
                                                                                                  predic_borden.loc[index,'Cyclo_lon'],
                                                                                                  predic_borden.loc[index,'xmin'],
                                                                                                  predic_borden.loc[index,'xmax'],
                                                                                                  predic_borden.loc[index,'ymin'],
                                                                                                  predic_borden.loc[index,'ymax'],
                                                                                                  predic_borden.loc[index,'Cyclo_yaw'],
                                                                                                  heights_table)
    except:
        print(metafile)
print(len(predic_borden))
predic_borden.to_csv(annotpath / 'predic_borden_with_results_stuk_amsterdam_07.csv')
print(len(predic_borden.dropna()))
sys.exit()
predic_borden = predic_borden.dropna()
annot_borden = pd.read_csv(str(annotpath / 'annot_borden_valid_db.csv'),index_col='borden_id')
print(annot_borden.head())

# True positive: predic_bord is binnen 5 meter van annot_bord van dezelfde klasse
# False positive: predic_bord is buiten 5 meter van annot_bord van dezelfde klasse
# False negative: annot_borden die niet gevonden worden
# True negative: 

# Set all predic_borden to 'False positive'. They will become 'True positive' if they match with an annot_bord.
predic_borden['result'] = 'False positive'

# Set all annot_borden to 'False negative'. They will become 'True positive' if they match with an predic_bord.
annot_borden['result'] = 'False negative'

predic_borden = predic_borden[['Cyclo_id','Class','bord_lat','bord_lon','result']]

for index_predic in predic_borden.index:
    coord_predic = (predic_borden.loc[index_predic,'bord_lat'],predic_borden.loc[index_predic,'bord_lon'])
    for index_annot in annot_borden.index:
        coord_annot = (annot_borden.loc[index_annot,'latitude'],annot_borden.loc[index_annot,'longitude'])
        distance = geopy.distance.distance(coord_predic, coord_annot).m
        if distance <=10 and annot_borden.loc[index_annot,'class']==predic_borden.loc[index_predic,'Class']:
            predic_borden.loc[index_predic,'result']='True positive'
            annot_borden.loc[index_annot,'result']='True positive'

annot_borden.to_csv(annotpath / 'valid_annot_borden_with_results.csv')
predic_borden.to_csv(annotpath / 'valid_predic_borden_with_results.csv')

print('Total tested: ',len(predic_borden))
print('True positives: ',len(predic_borden[predic_borden['result']=='True positive']))
print('False positives: ',len(predic_borden[predic_borden['result']=='False positive']))
print('False negatives: ',len(annot_borden[annot_borden['result']=='False negative']))
