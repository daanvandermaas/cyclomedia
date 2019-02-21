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

# Variable list:
path = Path(variables.mainpath) # Main directory
input_name_pointf_csv = 'stuk_amsterdam/latlon_amsterdam.csv' # Ouput from script 01
downloadDir = path / variables.cyclo_downloadDir
downloadDir = path / 'stuk_amsterdam/download'

# Cyclomedia url variables
baseurl = 'https://atlas.cyclomedia.com/PanoramaRendering/RenderByLocation2D/'
EPSG = '4326' # 4326 of 28992
width = '?width=1024'
height = '&height=512'
hfov = '&hfov=90'
index = '&index=%s'
apiKey = variables.cyclo_apiKey
xml_url = baseurl+EPSG+'/%s/%s'+width+height+hfov+index+apiKey+'&format=xml'
figure_url = baseurl+EPSG+'/%s/%s'+width+height+hfov+index+apiKey

# Import pandas pointdataframe
pointdf = pd.read_csv(str(path / input_name_pointf_csv), index_col=0,encoding = 'utf-8')
# pointdf = pointdf.sample(frac=1)

# Define multiprocessing function
def download_cycloramas(lon,lat,xml_url,figure_url,downloadDir):
    url = xml_url %(lon,lat,0)
    try:
        r=requests.get(xml_url %(lon,lat,0), auth=(variables.cyclo_username,variables.cyclo_pw))
        assert r.status_code==200
    
        # Split lat and lon of the cyclomedia picture
        for split in r.text.split():
            if split.startswith('recording-location-x='):
                lon_c = split.split('"')[1]
            if split.startswith('recording-location-y='):
                lat_c = split.split('"')[1]
            if split.startswith('recording-id='):
                this_cyclo_id = split.split('"')[1]
    
        # Check if that identifier is already downloaded, if not --> download
        for indexnr,indexletter in zip([0,1,2],['L','C','R']):
            url = figure_url %(lon,lat,indexnr)
            file = 'Cyclo_'+this_cyclo_id+'_'+indexletter+'_'+str(lon)+'_'+str(lat)+'.jpg'
            filepath = downloadDir / file
            if not filepath.is_file():
                #print('Download %s_%s' %(this_cyclo_id,indexletter))
                r=requests.get(url, auth=(variables.cyclo_username,variables.cyclo_pw))
                assert r.status_code==200
    
                with open(str(filepath), 'wb') as f_out:
                    r.raw.decode_content = True
                    for chunk in r.iter_content(1024):
                        f_out.write(chunk)
    
            r=requests.get(xml_url %(lon,lat,indexnr), auth=(variables.cyclo_username,variables.cyclo_pw))
            assert r.status_code==200
    
            textfilename = downloadDir / str(this_cyclo_id + '_' + indexletter + '.txt')
            if not textfilename.is_file():
                textfilename = str(textfilename)
                with open(textfilename, "w") as text_file:
                    for split in r.text.split():
                        print(split, file=text_file)
    #             else:
                    # Is already downloaded
                    # print('%s_%s is already downloaded' %(this_cyclo_id,indexletter))
    except:
        print('Could not download lon: %s lat %s' %(lon,lat))

# Multicore
# get max number of cores
nr_of_cores = mp.cpu_count()
use_amount_cores = int(np.rint(0.95*nr_of_cores) -1)
print('Using ',use_amount_cores, ' cores...')

# let the multicore magic distribute the function to all the cores 
a = datetime.datetime.now()
pool = mp.Pool(processes = use_amount_cores) # zijn er nog 5 over voor andere dingen:P
pool.starmap(download_cycloramas,zip(pointdf['lon'].values,pointdf['lat'].values,repeat(xml_url),repeat(figure_url),repeat(downloadDir)))
b = datetime.datetime.now()
c = b - a
print('Done in seconds: ', c.total_seconds())
