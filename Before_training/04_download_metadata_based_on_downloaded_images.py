import os,sys
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import multiprocessing as mp
from itertools import repeat
import datetime
import variables
import geopy
import geopy.distance
import random
from pyproj import Proj, transform

# Variable list:
path = Path('/flashblade/lars_data/2018_Cyclomedia_panoramas/project_folder/YOLO/borden_train_balanced')
downloadDir = path / 'metadata2'
inputDir = path / 'valid_img'

lons_4326 = []
lons_28992 = []
lats_4326 = []
lats_28992 = []

filelist = os.listdir(str(inputDir))
for file in filelist:
    if file.endswith('jpg'):
        split = file.split('_')
#        if split[2]== 'L':
        if float(split[-2]) < 90:
            print(split)
            lons_4326 = np.append(lons_4326,float(split[-2]))
            lats_4326 = np.append(lats_4326,float(split[-1].replace('.jpg','')))
        else:
            lons_28992 = np.append(lons_28992,float(split[-2]))
            lats_28992 = np.append(lats_28992,float(split[-1].replace('.jpg','')))

inProj = Proj(init='epsg:28992')
outProj = Proj(init='epsg:4326')
for x,y in zip(lons_28992,lats_28992):
    x2,y2 = transform(inProj,outProj,x,y)
    lons_4326 = np.append(lons_4326,x2)
    lats_4326 = np.append(lats_4326,y2)

    
# Cyclomedia url variables
baseurl = 'https://atlas.cyclomedia.com/PanoramaRendering/RenderByLocation2D/'
EPSG = '4326' # 4326 of 28992
width = '?width=1024'
height = '&height=512'
hfov = '&hfov=90'
index = '&index=%s'
apiKey = variables.cyclo_apiKey
xml_url = baseurl+EPSG+'/%s/%s'+width+height+hfov+index+apiKey+'&format=xml'

def download_metadata(lon,lat,xml_url,downloadDir):
    # First for 'L', then for 'C' and 'R'. The last get the same basename as 'L'
    url = xml_url %(lon,lat,0)
    r=requests.get(url, auth=(variables.cyclo_username,variables.cyclo_pw))
    try:
        assert r.status_code==200
    
        # textfilename = downloadDir / 
        for split in r.text.split():
            if split.startswith('recording-id='):
                this_cyclo_id = split.split('"')[1]
                #print(this_cyclo_id)
                break
        textfilename = downloadDir / str(this_cyclo_id + '_L.txt')
        if not textfilename.is_file():
            with open(str(textfilename), "w") as text_file:
                for split in r.text.split():
                    print(split, file=text_file)
    
        for indexnr,indexletter in zip([1,2],['C','R']):
            url = xml_url %(lon,lat,indexnr)
            r=requests.get(url, auth=(variables.cyclo_username,variables.cyclo_pw))
            assert r.status_code==200
    
            textfilename = downloadDir / str(this_cyclo_id + '_' + indexletter + '.txt')
            if not textfilename.is_file():
                with open(str(textfilename), "w") as text_file:
                    for split in r.text.split():
                        print(split, file=text_file)
    except:
        print('could not download',lat,lon)
# Multicore
# get max number of cores
nr_of_cores = mp.cpu_count()
use_amount_cores = int(np.rint(0.95*nr_of_cores) -1)
print('Using ',use_amount_cores, ' cores...')

# let the multicore magic distribute the function to all the cores 
a = datetime.datetime.now()
pool = mp.Pool(processes = use_amount_cores) # zijn er nog 5 over voor andere dingen:P
pool.starmap(download_metadata,zip(lons_4326,lats_4326,repeat(xml_url),repeat(downloadDir)))
b = datetime.datetime.now()
c = b - a
print('Done in seconds: ', c.total_seconds())

