

import numpy as np



class LoadData:
    
    def load(name):
        
        if name == 'train_set':
            
            x = np.load('dataset/x_train.npy')
            y = np.load('dataset/y_train.npy')
            
            return_set = (x,y)
        
        if name == 'test_set':
            
            x = np.load('dataset/x_test.npy')
            y = np.load('dataset/y_test.npy')
            
            return_set = (x,y)
        
        return return_set