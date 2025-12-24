import cv2
import numpy as np
from pathlib import Path
import tarfile

# Contrast limited adaptive histogram equalization
class CLAHE:
    
    def __init__(self, cliplimit=5, thresh=30):
        self.cliplimit = cliplimit
        self.thresh = thresh
    
    def fit_transform(self, image):
        image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=self.cliplimit)
        final_img = clahe.apply(image_bw) + self.thresh
        return final_img
    
# augmentation
class Augment:
    
    def __init__(self):
        pass
    
    def fit_transform(self, image):                
        #vertical flip
        img_flip_ud = cv2.flip(image, 0)
        img_flip_ud =  cv2.resize(img_flip_ud,(256,256))
        #horizontal flip
        img_flip_lr = cv2.flip(image, 1)
        img_flip_lr = cv2.resize(img_flip_lr,(256,256))
        # rotate
        img_rotate_180 = cv2.rotate(image, cv2.ROTATE_180)
        img_rotate_180 = cv2.resize(img_rotate_180,(256,256))
        # resize
        resized_image = cv2.resize(image,(256,256))        
        return [img_flip_ud, img_flip_lr, img_rotate_180, resized_image]
    
    

class Pipeline:
    def __init__(self, estimators):
        self.est = estimators
        self.len = len(estimators)

        
        
    def fit_transform(self, images):
        output_image = []
        for image in images:                
            temp_image = image
            for name, obj in self.est:
                temp_image = obj.fit_transform(temp_image)
                
            output_image.append(temp_image)
        return np.reshape(np.array(output_image), (-1, 256,256))
        



class PrepDataset:

    def run(self):
 
        
        
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

        
                
        train_data = map(cv2.imread, trainpaths)   
        test_data = map(cv2.imread, testpaths)   
        
        preprocess = Pipeline([
            ('Image Enhancment', CLAHE(cliplimit=5, thresh=30)),
            ('Image Augmentation', Augment())
            ])
        
        preprocess__train_data = preprocess.fit_transform(train_data)
        preprocess__test_data = preprocess.fit_transform(test_data)
        
        train_labels = create_labels(trainlabel__paths)
        test_labels = create_labels(testlabel__paths)
        
        np.save('dataset/x_train.npy',preprocess__train_data, allow_pickle=False)
        np.save('dataset/x_test.npy',preprocess__test_data, allow_pickle=False)
        
        np.save('dataset/y_train.npy',train_labels, allow_pickle=False)
        np.save('dataset/y_test.npy',test_labels, allow_pickle=False)
        
        
