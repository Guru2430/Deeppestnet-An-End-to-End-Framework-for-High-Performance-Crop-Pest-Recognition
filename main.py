import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from preprocess import preprocessing, Augment
from preprocess import Pipeline
from train import train



def main():
    

    # read image root path
    image__root_path = Path('dataset/ip102_v1.1/images')
    
    
    # make list of images and anotations
    images_paths = list(image__root_path.glob('*.*'))
    
    
    image = map(plt.imread, images_paths)
    
    train_data = map(plt.imread, images_paths)
    
    
    train=False
    if train:  
        
        # preprocessign pipepline
        preprocess = Pipeline([
            ('Image Enhancment', preprocessing(cliplimit=5, thresh=30)),
            ('Image Augmentation', Augment())
            ])
        
        preprocessed_image = preprocess.fit_transform(train_data)
        
        # model params
        params = {
            'width_per_group':3,
            'bottleneck_ratio':4,
            'num_blocks':8,
            'units':4,
            'shuffle_buffer_size':100,
            'batch_size':8, 
            'epochs':300
            }
        
        # training
    
        train(params['width_per_group'], params['bottleneck_ratio'],
              params['num_blocks'],
              params['units'], params['shuffle_buffer_size'], params['batch_size'],
              params['epochs'])
    
    # testing
    import test        
    import show
    import train_test

if __name__ == "__main__":
    main()
