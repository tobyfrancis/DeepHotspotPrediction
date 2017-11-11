import numpy as np
import h5py
import crystallography as xtal
import pandas as pd
import copy

def get_stats(dataset,custom_size=False):
    d = min(dataset.shape[0],dataset.shape[1],dataset.shape[2])
    a = int(2*np.sqrt(d**2/12))
    ''' this is a really strict limit, for fitting a circumscribed cube 
        that can be arbitrarily rotated, we should derive something less strict
        for rotations around the z-axis '''
    if custom_size and custom_size <= a:
        a = custom_size
    elif custom_size and custom_size > a:
        print('Custom Size too big to ensure rotations will fit in dataset, sizing down.')
    else:
        pass

    cube_shape = np.array([a,a,a])
    index_array = np.array(np.unravel_index(np.arange(a**3),cube_shape)).reshape(a**3,3)
    small_center = np.array(cube_shape/2).reshape(3,1)
    big_center = np.matrix([dataset.shape[0]/2,dataset.shape[1]/2,dataset.shape[2]/2])
    return [dataset.shape,small_center,big_center,index_array,cube_shape]

def batch(dataset):
    shape,small_center,big_center,index_array,shape = get_stats(dataset,custom_size=custom_size)
    index_array = index_array - small_center.flatten()
    index_array = np.matrix(index_array)
    data_shape = np.array([dataset.shape[0],dataset.shape[1],dataset.shape[2]])
    centering = (data_shape - min(data_shape))/2
    def load_batch(batch_size):
        batch = np.zeros((batch_size,shape[0],shape[1],shape[2],4))
        rotation_matrices = xtal.cu2om(xtal.randomOrientations(batch_size))
        ''' replace this rotation_matrices with what works for you '''
        for i in range(batch_size):
            big_center_copy = copy.deepcopy(big_center)
            rot_matrix = rotation_matrices[i]
            hkl = tuple(np.array((big_center_copy +\
                       (index_array*rot_matrix)),dtype=int).T)
            batch[i] = dataset[hkl].reshape((shape[0],shape[1],shape[2],4))
        return np.array(batch,dtype='float32')
return load_batch
