import numpy as np
import os
import argparse
import time

from preprocessing_utils import load_files, get_planes, get_crop

crop_size = (32,32)
n_crops = 2


parser = argparse.ArgumentParser()
parser.add_argument("--source", "-s", default="*", type=str, help='Source .npy file name inside clear(noised)_files directory')
parser.add_argument("--dir_name", "-d", default="../datasets", type=str, help='Directory path to datasets')

def get_planes_and_dump(source, dir_name):
    path_clear = os.path.join(dir_name,"clear_events", source)
    path_noise = os.path.join(dir_name,"noised_events", source)
    
    clear_readout_planes = []
    noised_readout_planes = []
    clear_collection_planes = []
    noised_collection_planes = []

    for file_clear, file_noise in load_files(path_clear, path_noise):
        a, b, c, d = get_planes(file_clear, file_noise)
        clear_readout_planes.append(a)
        noised_readout_planes.append(b)
        clear_collection_planes.append(c)
        noised_collection_planes.append(d)

    clear_readout_planes = np.concatenate(clear_readout_planes)
    noised_readout_planes = np.concatenate(noised_readout_planes)
    clear_collection_planes = np.concatenate(clear_collection_planes)
    noised_collection_planes = np.concatenate(noised_collection_planes)

    s = len(clear_readout_planes)
    p = np.random.permutation(s)
    c_r = clear_readout_planes[p]
    n_r = noised_readout_planes[p]

    np.save(os.path.join(dir_name,"clear_planes", 'readout_train'), c_r[:int(s*0.6)])
    np.save(os.path.join(dir_name,"clear_planes", 'readout_val'), c_r[int(s*0.6):int(s*0.8)])
    np.save(os.path.join(dir_name,"clear_planes", 'readout_test'), c_r[int(s*0.8):])

    np.save(os.path.join(dir_name,"noised_planes", 'readout_train'), n_r[:int(s*0.6)])
    np.save(os.path.join(dir_name,"noised_planes", 'readout_val'), n_r[int(s*0.6):int(s*0.8)])
    np.save(os.path.join(dir_name,"noised_planes", 'readout_test'), n_r[int(s*0.8):])


    s = len(clear_collection_planes)
    p = np.random.permutation(s)
    c_c = clear_collection_planes[p]
    n_c = noised_collection_planes[p]

    np.save(os.path.join(dir_name,"clear_planes", 'collection_train'), c_c[:int(s*0.6)])
    np.save(os.path.join(dir_name,"clear_planes", 'collection_val'), c_c[int(s*0.6):int(s*0.8)])
    np.save(os.path.join(dir_name,"clear_planes", 'collection_test'), c_c[int(s*0.8):])

    np.save(os.path.join(dir_name,"noised_planes", 'collection_train'), n_c[:int(s*0.6)])
    np.save(os.path.join(dir_name,"noised_planes", 'collection_val'), n_c[int(s*0.6):int(s*0.8)])
    np.save(os.path.join(dir_name,"noised_planes", 'collection_test'), n_c[int(s*0.8):])
    #print(c_r.shape)
    #print(n_r.shape)
    #print(c_c.shape)
    #print(n_c.shape)

def crop_planes_and_dump(dir_name):
    for s in ['readout_', 'collection_']:
        for ss in ['train.npy', 'val.npy', 'test.npy']:
            clear_crops = []
            noised_crops = []
            clear_plane = np.load(os.path.join(dir_name,"clear_planes", s+ss))
            noised_plane = np.load(os.path.join(dir_name,"noised_planes", s+ss))

            for clear_crop, noised_crop in get_crop(clear_plane, noised_plane, n_crops, crop_size):
                clear_crops.append(clear_crop)
                noised_crops.append(noised_crop)
            clear_crops = np.concatenate(clear_crops)
            noised_crops = np.concatenate(noised_crops)
            np.save(os.path.join(dir_name,"clear_crops", s+ss), clear_crops)
            np.save(os.path.join(dir_name,"noised_crops", s+ss), noised_crops)
            #print(clear_crops.shape, noised_crop.shape)


def main(source, dir_name):
    
    for i in ['clear_planes', 'clear_crops', 'noised_planes', 'noised_crops']:
        if not os.path.isdir(os.path.join(dir_name,i)):
            os.mkdir(os.path.join(dir_name,i))

    get_planes_and_dump(source, dir_name)
    for s in ['readout_', 'collection_']:
        for ss in ['train.npy', 'val.npy', 'test.npy']:
            print(s+ss + ' clear', np.load(os.path.join(dir_name, 'clear_planes', s+ss)).shape)
            print(s+ss + ' noised', np.load(os.path.join(dir_name, 'noised_planes', s+ss)).shape)
    


    crop_planes_and_dump(dir_name)
    for s in ['readout_', 'collection_']:
        for ss in ['train.npy', 'val.npy', 'test.npy']:
            print(s+ss + ' clear', np.load(os.path.join(dir_name, 'clear_crops', s+ss)).shape)
            print(s+ss + ' noised', np.load(os.path.join(dir_name, 'noised_crops', s+ss)).shape)

    

if __name__ == '__main__':
    args = vars(parser.parse_args())
    start = time.time()
    main(**args)
    print('Program done in %f'%(time.time()-start))
