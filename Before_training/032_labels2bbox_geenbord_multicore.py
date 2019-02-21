#%%

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys,os
import variables
from scipy.ndimage.measurements import label
import multiprocessing as mp
from itertools import repeat
import datetime

# -- converts a mask to an array of bounding boxes
def mask2box(mask, classes):
    # -- the structure defining connectedness    
    structure = np.ones((3, 3), dtype=np.int)  # this defines the connection filter
    # -- create an object to write the bounding boxes to
    bbox = list()

    # -- loop through all classes
    for c in classes:
        # -- remove all labels that do not correspond to c and label the connected components of c 1 to n
         mask_class = mask.copy()
         mask_class[mask_class != c] = 0
         mask_class[mask_class == c] = 1
         labeled, ncomponents = label(mask_class, structure)

         for n in range(1,ncomponents+1):
             coords = np.where(labeled == n)
             bbox.append([c, min(coords[1]), max(coords[1]), min(coords[0]), max(coords[0]) ])
    return(bbox)
    
def renameandcheck(output,classes_names):
    sizes = []
    for t in output:
        for i in np.arange(len(classes_names)):
            if t[0] == i+1:
                t[0] = classes_names[i]
    return output

def writebbox2xml(output,outputDir,file):
    # --  Write output of boundixbox to xml file
    outputPath = outputDir / file.replace('.txt', '.xml')
    with open(str(outputPath), 'w') as the_file:
        the_file.write('<annotation verified="yes">')
        the_file.write('\n\t<folder>images</folder>')
        the_file.write('\n\t<filename>%s</filename>' %(file.replace('.txt','.jpg')))
        the_file.write('\n\t<path>/flashblade/lars_data/2018_Cyclomedia_panoramas/project_folder/YOLO/borden_train/img/%s </path>'%(file.replace('.txt','.jpg'))) # MAG NIET LEEG ZIJN
        the_file.write('\n\t<source>')
        the_file.write('\n\t\t<database>Unknown</database>')
        the_file.write('\n\t</source>')
        the_file.write('\n\t<size>')
        the_file.write('\n\t\t<width>1024</width>')
        the_file.write('\n\t\t<height>512</height>')
        the_file.write('\n\t\t<depth>3</depth>')
        the_file.write('\n\t</size>')
        the_file.write('\n\t<segmented>0</segmented>')
        for t in output:
            xsize = t[2]-t[1]
            ysize = t[4]-t[3]
            if xsize*ysize > 200:
                the_file.write('\n\t<object>')
                the_file.write('\n\t\t<name>%s</name>' %(t[0]))
                the_file.write('\n\t\t<pose>Unspecified</pose>')
                the_file.write('\n\t\t<truncated>0</truncated>')
                the_file.write('\n\t\t<difficult>0</difficult>')
                the_file.write('\n\t\t<bndbox>')
                the_file.write('\n\t\t\t<xmin>%s</xmin>'%(t[1]))
                the_file.write('\n\t\t\t<ymin>%s</ymin>'%(t[3]))
                the_file.write('\n\t\t\t<xmax>%s</xmax>'%(t[2]))
                the_file.write('\n\t\t\t<ymax>%s</ymax>'%(t[4]))
                the_file.write('\n\t\t</bndbox>')
                the_file.write('\n\t</object>')
        the_file.write('\n</annotation>')

path = Path('/flashblade/lars_data/2018_Cyclomedia_panoramas/project_folder/')

labelpath = path / 'BirdsAI_falsepositives/labels'
imgpath = path / 'BirdsAI_falsepositives/need_labeling'
outputpath = path / 'BirdsAI_falsepositives/outputdir'
classes = variables.classes_nr
classes_names = variables.classes_names

labelList = os.listdir(str(labelpath))
print(labelList)

def process_files(file,labelpath,imgpath,outputpath,classes,classes_names):
    if file.endswith('txt'):
        mask = np.genfromtxt(str(labelpath / file), delimiter= ',')
        try:    
            img = plt.imread(str(imgpath / file.replace('.txt','.jpg')))
            #plt.imsave(str(outputpath / file.replace('.txt','.jpg' )),np.rint(img).astype(np.uint8))
            if np.mean(mask) >= 0: # foto's met borden (dus als alle pixels als 0 zijn geclassificeerd)
                img = plt.imread(str(imgpath / file.replace('.txt','.jpg')))
                
                # -- Normal orientation
                output = mask2box(mask,classes)
                output = renameandcheck(output,classes_names)
                writebbox2xml(output,outputpath,file)
                plt.imsave(str(outputpath / file.replace('.txt','.jpg' )),np.rint(img).astype(np.uint8))
        except:
            print(file)

# Multicore
# get max number of cores
nr_of_cores = mp.cpu_count()
use_amount_cores = int(np.rint(0.95*nr_of_cores) -1)
print('Using ',use_amount_cores, ' cores...')

# let the multicore magic distribute the function to all the cores 
a = datetime.datetime.now()
pool = mp.Pool(processes = use_amount_cores) # zijn er nog 5 over voor andere dingen:P
pool.starmap(process_files,zip(labelList,repeat(labelpath),repeat(imgpath),repeat(outputpath),repeat(classes),repeat(classes_names)))
b = datetime.datetime.now()
c = b - a
print('Done in seconds: ', c.total_seconds())
