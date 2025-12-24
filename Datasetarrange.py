
# load data
import numpy as np
import pathlib
from pathlib import Path
import zipfile


class PrepDataset:

    def run(self):
        file_paths = {
            'classsification_path' :'Classification-20231127T063248Z-002.zip',
            'detection_path': 'Detection-20231127T064718Z-001.zip',
            'image_path': 'dataset/Detection/VOC2007/JPEGImages.tar',
            'annotation': 'dataset/Detection/VOC2007/Annotations.tar'
            }
        
            
        with zipfile.ZipFile(file_paths['classsification_path']) as zip_file:
            zip_file.extractall('dataset')
            
            
            
        with zipfile.ZipFile(file_paths['detection_path']) as zip_file:
            zip_file.extractall('dataset')
            
        
        
        import tarfile
        
        with tarfile.open(file_paths['image_path']) as image:
            image.extractall('dataset')
        
        with tarfile.open(file_paths['annotation']) as ano:
            ano.extractall('dataset')
        

        import xml.etree.ElementTree as ET
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
            
        
        def create_segment_image(annotation_path, image_path):
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            
            for i in root.find('object'):
                if i.tag == 'bndbox':
                    xmin = int(i[0].text)
                    ymin = int(i[1].text)
                    xmax = int(i[2].text)
                    ymax = int(i[3].text)
            
            
            image = plt.imread(image_path)
            plt.figure()
            plt.imshow(image)
            
            
            plt.gca().add_patch(Rectangle((xmin,ymin),xmax,ymax,
                                edgecolor='white',
                                facecolor='white',
                                lw=4))
            plt.axis('off')
            plt.savefig(Path('dataset/Segmentation/', image_path.stem+'.jpg'), 
                        bbox_inches='tight', pad_inches=0)
            plt.close()
            
            
            
 
        
        # read image root path
        image__root_path = pathlib.Path('dataset/JPEGImages')
        annotation__root_path = pathlib.Path('dataset/Annotations')
        
        
        # make list of images and anotations
        images_paths = list(image__root_path.glob('*.*'))
        annotation_paths = list(annotation__root_path.glob('*.*'))
        
        # create segmentation folder to save segmented image
        import os
        
        if os.path.exists('dataset/Segmentation'):
            pass
        else:
            os.mkdir('dataset/Segmentation')
        
        
        # save segment image
        
        seg = map(create_segment_image, annotation_paths, images_paths)
        
        for _ in seg:
            pass
        

        import tarfile
        
        with tarfile.open('dataset/Classification/ip102_v1.1.tar') as image:
            image.extractall('dataset')
        
        
        

        
        count=0
        trainpaths = []
        trainlabel__paths = []
        
        with open('dataset/ip102_v1.1/train.txt', 'r') as train:
            for line in train:
                print(line)
                split = line.split(' ')
                filename = split[0]
                label = int(split[-1])
                print(filename, label)
                
                if label in range(0,10):
                    
                    
                    path = Path('dataset/ip102_v1.1/images', filename)
                    trainpaths.append(path)
                    trainlabel__paths.append(label)
                    print(str(path))
                    count+=1

        
        testpaths = []
        testlabel__paths = []
        
        with open('dataset/ip102_v1.1/test.txt', 'r') as train:
            for line in train:
                split = line.split(' ')
                filename = split[0]
                label = int(split[-1])
        
                
                if label in range(0,10):
                    
                    
                    path = Path('dataset/ip102_v1.1/images', filename)
                    testpaths.append(path)
                    testlabel__paths.append(label)
                    print('*', end='')
        

        
        valpaths = []
        vallabel__paths = []
        
        with open('dataset/ip102_v1.1/test.txt', 'r') as train:
            for line in train:
                split = line.split(' ')
                filename = split[0]
                label = int(split[-1])
        
                
                if label in range(0,10):
                    
                    
                    path = Path('dataset/ip102_v1.1/images', filename)
                    valpaths.append(path)
                    vallabel__paths.append(label)
                    print('*', end='')
        

        
        def create_labels(testlabel__paths):
            new_labels = []
            
            for row in testlabel__paths:
                temp = np.ones(4, dtype=np.int32) * row
                new_labels.append(temp)
                
            new_labels = np.array(new_labels).reshape(-1)
