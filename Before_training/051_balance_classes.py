import xml.etree.ElementTree as ET
from pathlib import Path
import os,sys
import numpy as np
import random
from sklearn.model_selection import train_test_split

def parse_annotation(ann_dir, img_dir, labels=[]):
    all_imgs = []
    A01_100o_list = []
    A08_80_list = []
    A01_80_list = []
    G01_list = []
    J01_list = []
    J27_list = []
    A01_100s_list = []
    A01_100_list = []
    B06_list = []
    C02_list = []
    L05_list = []
    seen_labels = {}

    for ann in sorted(os.listdir(ann_dir)):
        if ann.endswith('xml'):
            print(os.path.join(ann_dir,ann))
            img = {'object':[]}
    
            tree = ET.parse(os.path.join(ann_dir,ann))
#            print(tree)
            for elem in tree.iter():
                if 'filename' in elem.tag:
                    img['filename'] = img_dir + elem.text
                if 'width' in elem.tag:
                    img['width'] = int(elem.text)
                if 'height' in elem.tag:
                    img['height'] = int(elem.text)
                if 'object' in elem.tag or 'part' in elem.tag:
                    obj = {}
    
                    for attr in list(elem):
                        if 'name' in attr.tag:
                            obj['name'] = attr.text
                            bord = obj['name']
                            print(bord.replace('-','_')) 
#labels = ['A01-100o','A08-80','A01-80','G01','J01','C02','J27','A01-100s','A01-100','B06']
                            if bord == 'A01-100o':
                                A01_100o_list = np.append(A01_100o_list,ann) 
                            if bord == 'A08-80':
                                A08_80_list = np.append(A08_80_list,ann) 
                            if bord == 'A01-80':
                               A01_80_list = np.append(A01_80_list,ann) 
                            if bord == 'G01':
                               G01_list = np.append(G01_list,ann) 
                            if bord == 'J01':
                               J01_list  = np.append(J01_list,ann) 
                            if bord == 'J27':
                               J27_list  = np.append(J27_list,ann) 
                            if bord == 'A01-100s':
                               A01_100s_list  = np.append(A01_100s_list,ann) 
                            if bord == 'A01-100':
                               A01_100_list  = np.append(A01_100_list,ann) 
                            if bord == 'B06':
                               B06_list  = np.append(B06_list,ann) 
                            if bord == 'C02':
                               C02_list  = np.append(C02_list,ann) 
                            if bord == 'L05':
                               L05_list  = np.append(L05_list,ann) 
                            if obj['name'] in seen_labels:
                                seen_labels[obj['name']] += 1
                            else:
                                seen_labels[obj['name']] = 1
    
                            if len(labels) > 0 and obj['name'] not in labels:
                                break
                            else:
                                img['object'] += [obj]
    
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
    
            if len(img['object']) > 0:
                all_imgs += [img]

    return all_imgs, seen_labels,A01_100o_list,A08_80_list,A01_80_list,G01_list,J01_list,J27_list,A01_100s_list,A01_100_list,B06_list,C02_list,L05_list

inputpath = Path('/flashblade/lars_data/2018_Cyclomedia_panoramas/project_folder/YOLO/borden_train_balanced')
annotation_dir = inputpath / 'valid_annot'
image_dir = inputpath / 'valid_img'
labels = ['A01-100o','A08-80','A01-80','G01','J01','C02','J27','A01-100s','A01-100','B06','L05']

outputpath = Path('/flashblade/lars_data/2018_Cyclomedia_panoramas/project_folder/YOLO/borden_train_balanced')
train_annot_dir = outputpath / 'annot'
train_img_dir = outputpath / 'img'
test_annot_dir = outputpath / 'valid_annot'
test_img_dir = outputpath / 'valid_img'

all_imgs, seen_labels,A01_100o_list,A08_80_list,A01_80_list,G01_list,J01_list,J27_list,A01_100s_list,A01_100_list,B06_list,C02_list,L05_list = parse_annotation(str(annotation_dir),str(image_dir),labels)
#print(all_imgs)
print(seen_labels)
sys.exit()
#{'G01': 2, 'A01-100s': 1, 'B06': 3, 'A01-80': 1, 'J01': 2, 'J27': 1, 'A08-80': 1, 'A01-100': 1, 'A01-100o': 1, 'C02': 1}
print('G01',len(G01_list),'A01-100s',len(A01_100s_list),'B06',len(B06_list),'A01-80',len(A01_80_list),'J01',len(J01_list),'J27',len(J27_list),'A08-80',len(A08_80_list),'A01-100',len(A01_100_list),'A01-100o',len(A01_100o_list),'C02',len(C02_list),'L05',len(L05_list))

def length(x):
    return len(x)

variable_list = [G01_list,A01_100s_list,B06_list,A01_80_list,J01_list,J27_list,A08_80_list,A01_100_list,A01_100o_list,C02_list,L05_list]

# Split the lists in test and train
split = 0.2
B06_train, B06_test = train_test_split(B06_list, test_size=split,random_state=42)
G01_train, G01_test = train_test_split(G01_list, test_size=split,random_state=42)
A01_100s_train, A01_100s_test = train_test_split(A01_100s_list, test_size=split,random_state=42)
A01_80_train, A01_80_test = train_test_split(A01_80_list, test_size=split,random_state=42)
J01_train, J01_test = train_test_split(J01_list, test_size=split,random_state=42)
J27_train, J27_test = train_test_split(J27_list, test_size=split,random_state=42)
A08_80_train, A08_80_test = train_test_split(A08_80_list, test_size=split,random_state=42)
A01_100_train, A01_100_test = train_test_split(A01_100_list, test_size=split,random_state=42)
A01_100o_train, A01_100o_test = train_test_split(A01_100o_list, test_size=split,random_state=42)
C02_train, C02_test = train_test_split(C02_list, test_size=split,random_state=42)
L05_train, L05_test = train_test_split(L05_list, test_size=split,random_state=42)

# Remove signs from the set that also occur in the train set
all_train = np.concatenate((B06_train,G01_train,A01_100s_train,A01_80_train,J01_train,J27_train,A08_80_train,A01_100_train,A01_100o_train,C02_train,L05_train))

var_list = [G01_test,A01_100s_test,B06_test,A01_80_test,J01_test,J27_test,A08_80_test,A01_100_test,A01_100o_test,C02_test,L05_test]
def remove_train_from_test(test,train):
    keep = []
    for i in np.arange(len(test)):
        if test[i] in train:
            pass
        else:
            keep = np.append(keep,test[i])
    return keep
G01_test,A01_100s_test,B06_test,A01_80_test,J01_test,J27_test,A08_80_test,A01_100_test,A01_100o_test,C02_test,L05_test = (remove_train_from_test(var,all_train) for var in var_list)

all_test = np.concatenate((B06_test,G01_test,A01_100s_test,A01_80_test,J01_test,J27_test,A08_80_test,A01_100_test,A01_100o_test,C02_test,L05_test))

# Check if there are any duplicates remaining
i=0
for file in all_test:
    if file in all_train:
        print(file)
        i = i+1
print(i)

### Train
# Find longest dataset
max_length = max(len(B06_train),len(G01_train),len(A01_100s_train),len(A01_80_train),len(J01_train),len(J27_train),len(A08_80_train),len(A01_100_train),len(A01_100o_train),len(C02_train),len(L05_train))
print(max_length)

# Add random signs to the datasets to get them to equal length
var_list = [G01_train,A01_100s_train,B06_train,A01_80_train,J01_train,J27_train,A08_80_train,A01_100_train,A01_100o_train,C02_train,L05_train]

def add_random_files(var,max_length):
    print(len(var),'before')
    nr_to_add = max_length - len(var)
    to_add = []
    for i in np.arange(nr_to_add):
        print(i)
        to_add = np.append(to_add,random.choice(var))
    var = np.append(var,to_add)
    print(var,len(var),'after')
    return var


G01_train,A01_100s_train,B06_train,A01_80_train,J01_train,J27_train,A08_80_train,A01_100_train,A01_100o_train,C02_train,L05_train = (add_random_files(var,max_length) for var in var_list)
var_list = [G01_train,A01_100s_train,B06_train,A01_80_train,J01_train,J27_train,A08_80_train,A01_100_train,A01_100o_train,C02_train,L05_train]

print(J27_train)
print(min(len(B06_train),len(G01_train),len(A01_100s_train),len(A01_80_train),len(J01_train),len(J27_train),len(A08_80_train),len(A01_100_train),len(A01_100o_train),len(C02_train),len(L05_train)))

for var in var_list:
    for file in var:
        print(file)
        if os.path.isfile(str(train_img_dir / file.replace('xml','jpg'))) == False:
            command = 'ln %s %s' %(str(image_dir / file.replace('xml','jpg')), str(train_img_dir / file.replace('xml','jpg')))
            os.system(command)
        prefix = 0
        i=False
        while i == False:
            filename = str(prefix) + '_' + file
            if os.path.isfile(str(train_annot_dir / filename)):
                prefix = prefix+1
            else:
                command = 'ln %s %s' %(str(annotation_dir / file), str(train_annot_dir / filename))
                os.system(command)
                i=True
### TEST
# Find longest dataset
max_length = max(len(B06_test),len(G01_test),len(A01_100s_test),len(A01_80_test),len(J01_test),len(J27_test),len(A08_80_test),len(A01_100_test),len(A01_100o_test),len(C02_test),len(L05_test))
print(max_length)

# Add random signs to the datasets to get them to equal length
var_list = [G01_test,A01_100s_test,B06_test,A01_80_test,J01_test,J27_test,A08_80_test,A01_100_test,A01_100o_test,C02_test,L05_test]

def add_random_files(var,max_length):
    print(len(var),'before')
    nr_to_add = max_length - len(var)
    to_add = []
    for i in np.arange(nr_to_add):
        print(i)
        to_add = np.append(to_add,random.choice(var))
    var = np.append(var,to_add)
    print(var,len(var),'after')
    return var


G01_test,A01_100s_test,B06_test,A01_80_test,J01_test,J27_test,A08_80_test,A01_100_test,A01_100o_test,C02_test,L05_test = (add_random_files(var,max_length) for var in var_list)
var_list = [G01_test,A01_100s_test,B06_test,A01_80_test,J01_test,J27_test,A08_80_test,A01_100_test,A01_100o_test,C02_test,L05_test]

print(J27_test)
print(min(len(B06_test),len(G01_test),len(A01_100s_test),len(A01_80_test),len(J01_test),len(J27_test),len(A08_80_test),len(A01_100_test),len(A01_100o_test),len(C02_test),len(L05_test)))

for var in var_list:
    for file in var:
        print(file)
        if os.path.isfile(str(test_img_dir / file.replace('xml','jpg'))) == False:
            command = 'ln %s %s' %(str(image_dir / file.replace('xml','jpg')), str(test_img_dir / file.replace('xml','jpg')))
            os.system(command)
        prefix = 0
        i=False
        while i == False:
            filename = str(prefix) + '_' + file
            if os.path.isfile(str(test_annot_dir / filename)):
                prefix = prefix+1
            else:
                command = 'ln %s %s' %(str(annotation_dir / file), str(test_annot_dir / filename))
                os.system(command)
                i=True
