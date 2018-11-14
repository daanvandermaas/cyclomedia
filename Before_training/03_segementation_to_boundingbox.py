#%%
import numpy as np
from scipy.ndimage.measurements import label
from pathlib import Path
import os,sys
import variables

path = Path(variables.mainpath)
labelDir = path / variables.BAI_labelDir
outputDir = path / variables.bboxDir
classes = variables.classes_nr
classes_names = variables.classes_names

fileList = os.listdir(str(labelDir))

for file in fileList: 
    mask = np.genfromtxt(labelDir / file, delimiter= ',')

    #converts a mask to an array of bounding boxes
    def mask2box(mask, classes):
        #the sturcute defningn connectedness    
        structure = np.ones((3, 3), dtype=np.int)  # this defines the connection filter
        #create an object to write the bounding boxes to
        bbox = list()

        #loop through all classes
        for c in classes:
            #remove all labels that do not correspond to c and label the connected components of c 1 to n
             mask_class = mask.copy()
             mask_class[mask_class != c] = 0
             mask_class[mask_class == c] = 1
             labeled, ncomponents = label(mask_class, structure)

             for n in range(1,ncomponents+1):
                 coords = np.where(labeled == n)
                 bbox.append([c, min(coords[1]), max(coords[1]), min(coords[0]), max(coords[0]) ])

        return(bbox)

    output = mask2box(mask,classes)
    print(output)
    for t in output:
        print(t)
        if t[0] == 1:
            t[0] = classes_names[0]
        if t[0] == 2:
            t[0] = classes_names[1]
        if t[0] == 3:
            t[0] = classes_names[2]


    # Write output of boundixbox to xml file
    outputPath = outputDir / file.replace('.txt', '.xml')
    with open(str(outputPath), 'w') as the_file:
        the_file.write('<annotation verified="yes">')
        the_file.write('\n\t<folder>images</folder>')
        the_file.write('\n\t<filename>%s</filename>' %(file.replace('.txt', '.jpg')))
        the_file.write('\n\t<path>/Users/datitran/Desktop/raccoon/images/raccoon-177.jpg</path>') # MAG NIET LEEG ZIJN
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
