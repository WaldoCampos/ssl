import tensorflow_datasets as tfds
import tfds_qd.tfds_qd as tfds_qd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import argparse

def view(ds, n_rows, n_cols) :    
    _, ax = plt.subplots(n_rows, n_cols)
    for i in range(n_rows) :
        for j in range(n_cols) :
            ax[i,j].set_axis_off()
            
    for batch in ds :                           
        for i, sample in enumerate(batch):                    
            sketch = sample.numpy().astype(np.uint8)
            ax[i // n_cols][i % n_cols].imshow(sketch, cmap  = 'gray')
            min_v=np.min(sketch)
            max_v=np.max(sketch)
            print('minmax {} {}'.format(min_v, max_v))
            
        plt.waitforbuttonpress(1)
    plt.show()


def map_func(image):
    image = image['image']    
    return image

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type = str, required = True)    
    parser.add_argument('-data', type = str, required = True)      
    parser.add_argument('-dir', type = str, required = False)
    args = parser.parse_args()
    dataset = args.dataset    
    data = args.data
    data_dir ='~/tensorflow_datasets/'
    if not dir is None :
        data_dir = args.dir    
    ds = tfds.load(dataset, data_dir = data_dir)
    ds_test = ds[data].map(map_func).shuffle(1024).batch(25)        
    #ds_test = ds_test.take(10)
    view(ds_test, 5, 5)
