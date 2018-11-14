
# coding: utf-8

# In[2]:


import geopandas as gpd
import fiona
from shapely.geometry import shape, mapping
import shapely as sp
import os,sys
from pathlib import Path
import numpy as np
import pandas as pd
from unidecode import unidecode
import variables

# variable list:
path = Path(variables.mainpath) # Main directory
nwb_dir = variables.nwb_dir # Hier worden de xy coordinaten van de shapefile uitgehaald
nwb_name = variables.nwb_name
output_dir = variables.output_dir
name_cutout_shapefile = variables.name_cutout_shapefile # Naam van de uitsnede van het nwb
interpolate_per_x_m = variables.interpolate_per_x_m
name_point_shapefile = variables.name_point_shapefile # shp bestand met daarin punten op de locaties van wegen
output_name_pointf_csv = variables.output_name_pointf_csv # csv bestand om in het volgende bestand door te geven
# Of we krijgen een lijn of een polygoon waarvan we dan zelf uitknippen wat we willen.

# Hoek coordinaten van het vierkant in RD (Bij gebrek aan shapefile)
xmin = 136097
xmax = 136665
ymin = 457835
ymax = 458358

# Read NWB
nwb = gpd.read_file(str(path / nwb_dir / nwb_name))

cutout = nwb.cx[xmin:xmax,ymin:ymax]

cutout.to_file(driver = 'ESRI Shapefile', filename= str(path / output_dir / name_cutout_shapefile))

# creation of the resulting shapefile
schema = {'geometry': 'Point','properties': {'id': 'int'}}
crs = cutout.crs

with fiona.open(str(path / output_dir / name_point_shapefile), 'w', 'ESRI Shapefile', schema, crs=crs) as output:
    pointlistx = []
    pointlisty = []
    pointliststreet = []
    for i in np.arange(len(cutout)):
        street = cutout.iloc[i,12]
        geom = cutout.iloc[i,-1]
        print(street,geom)

        # length of the LineString
        length = geom.length

        # creation of the resulting shapefile
        schema = {'geometry': 'Point','properties': {'id': 'int'}}
        # create points every 10 meters along the line
        for i, distance in enumerate(range(0, int(length), interpolate_per_x_m)):
            point = geom.interpolate(distance)   
            output.write({'geometry':mapping(point),'properties': {'id':i}})
#             print(point.bounds)
            pointlistx = np.append(pointlistx,point.bounds[0])
            pointlisty = np.append(pointlisty,point.bounds[0])
            pointliststreet = np.append(pointliststreet,street)

#print(pointliststreet,pointlistx,pointlisty)

## Translate special characters:
#for i in np.arange(len(pointliststreet)):
#    temp=unidecode(pointliststreet[i])
#    pointliststreet[i]=temp

pointdf = pd.DataFrame({
    'lon' : pointlistx,
    'lat' : pointlisty,
    'street': pointliststreet
})

pointdf.to_csv(str(path /output_dir/ output_name_pointf_csv),encoding='utf-8')

