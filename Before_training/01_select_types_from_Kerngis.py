#%%

import geopandas as gpd
import os,sys
from pathlib import Path
import numpy as np
import pandas as pd

path = Path('/flashblade/lars_data/2018_Cyclomedia_panoramas/project_folder/input/punten/')
types = ['A01-80','A01-100','A01-100-S','A04-80','B06','G01','J02','J27','L5','L04-O-A-A']


#punten = gpd.read_file(str(path / 'Utrecht-Punten.shp'))
punten = gpd.read_file(str(path / 'Kerngis_NL.shp'))


#%%
print(punten['Type'].value_counts())
#print(punten[(punten['omschr']=='verkeersbord zonder kabel')])

soortlist = []
lonlist = []
latlist = []
for Type in types:
	borden = punten[(punten['Type'].str.contains(Type))]
	for index in borden.index:
		soortlist = np.append(soortlist,borden.loc[index,'Type'])
		point = (borden.loc[index,'geometry'])
		lonlist = np.append(lonlist,point.bounds[0])
		latlist = np.append(latlist,point.bounds[1])

latlondf = pd.DataFrame({'lon': lonlist,'lat':latlist,'soort':soortlist})
latlondf.to_csv('latlon.csv')
