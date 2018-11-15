#%%

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys,os
import variables
from scipy.ndimage.measurements import label
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from skimage.transform import rotate
import random
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
    
def rename(output,classes_names):
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

labelpath = path / 'BirdsAI/accepted'
imgpath = path / 'downloaded_panoramas_shapefile/met_bord'
outputpath = path / 'BirdsAI/outputdir'
classes = variables.classes_nr
classes_names = variables.classes_names
stringsToCheck = ['.txt'] # De borden die data augmentatie nodig hebben. 

labelList = os.listdir(str(labelpath))
print(labelList)

def process_files(file,labelpath,imgpath,outputpath,classes,classes_names,stringsToCheck):
    if file.endswith('txt'):
        mask = np.genfromtxt(str(labelpath / file), delimiter= ',')
        img = plt.imread(str(imgpath / file.replace('.txt','.jpg')))
        plt.imsave(str(outputpath / file.replace('.txt','.jpg' )),np.rint(img).astype(np.uint8))
        if np.mean(mask) > 0: # foto's zonder borden (dus als alle pixels als 0 zijn geclassificeerd hoeft het plaatje niet meegetraind te worden)
            img = plt.imread(str(imgpath / file.replace('.txt','.jpg')))
            
            # -- Normal orientation
            output = mask2box(mask,classes)
            output = rename(output,classes_names)
            writebbox2xml(output,outputpath,file)
            plt.imsave(str(outputpath / file.replace('.txt','.jpg' )),np.rint(img).astype(np.uint8))
            
            # -- Check if data_aug is necessary
            if any(string in file for string in stringsToCheck):
                data_aug = True
            else:
                data_aug = False
                
            # -- Data augmentation:
            if data_aug:
                # -- rotate
                nr_rotations = 6
                for i in np.arange(nr_rotations):
                    angle = random.choice([-15,-10,-5,5,10,15])
                    img_rot = rotate(img, angle=angle, mode='reflect')                
                    mask_rot = rotate(mask, angle=angle, mode='reflect') 
                    output = mask2box(mask_rot,classes)
                    output = rename(output,classes_names)
                    outputfile = file.replace('.txt','_rot_%s.txt' %(i))
                    writebbox2xml(output,outputpath,outputfile)
                    plt.imsave(str(outputpath / outputfile.replace('.txt','.jpg' )),img_rot)
                    
                # -- Gaussian random noise 
                noise = np.random.normal(0,25,(512,1024,3))
                img_noise = img+noise
                img_noise[img_noise>255] = 255
                img_noise[img_noise<0] = 0
                output = mask2box(mask,classes)
                output = rename(output,classes_names)
                outputfile = file.replace('.txt','_gn.txt')
                writebbox2xml(output,outputpath,outputfile)
                plt.imsave(str(outputpath / outputfile.replace('.txt','.jpg' )),np.rint(img_noise).astype(np.uint8))
                
                # -- shift left-right
                nr_of_shifts = 4
                choice = np.arange(-100,100,1)
                choice = choice[(choice>10) | (choice<-10)]
                for i in np.arange(nr_of_shifts):
                    shift = random.choice(choice)
                    img_rolled = np.roll(img,shift,axis=1)
                    mask_rolled = np.roll(mask,shift,axis=1)
                    output = mask2box(mask_rolled,classes)
                    output = rename(output,classes_names)
                    outputfile = file.replace('.txt','_shift_%s.txt' %(i))
                    writebbox2xml(output,outputpath,outputfile)
                    plt.imsave(str(outputpath / outputfile.replace('.txt','.jpg' )),img_rolled)

# Multicore
# get max number of cores
nr_of_cores = mp.cpu_count()
use_amount_cores = int(np.rint(0.95*nr_of_cores) -1)
print('Using ',use_amount_cores, ' cores...')

# let the multicore magic distribute the function to all the cores 
a = datetime.datetime.now()
pool = mp.Pool(processes = use_amount_cores) # zijn er nog 5 over voor andere dingen:P
pool.starmap(process_files,zip(labelList,repeat(labelpath),repeat(imgpath),repeat(outputpath),repeat(classes),repeat(classes_names),repeat(stringsToCheck)))
b = datetime.datetime.now()
c = b - a
print('Done in seconds: ', c.total_seconds())
