import numpy as np
import os
import argparse
import time

from preprocessing_utils import load_files, get_planes, get_crop


parser = argparse.ArgumentParser()
parser.add_argument("--source", "-s", default="*", type=str, help='Source .npy file name inside clear(noised)_files directory')
parser.add_argument("--dir_name", "-d", default="../datasets", type=str, help='Directory path to datasets')

def get_planes_and_dump(source, dir_name):
    path_clear = os.path.join(dir_name,"clear_events", source)
    path_noise = os.path.join(dir_name,"noised_events", source)
    
    clear_planes = []
    noised_planes = []    
    
    for file_clear, file_noise in load_files(path_clear, path_noise):
        for plane_clear, plane_noise in get_planes(file_clear, file_noise):
            clear_planes.append(plane_clear)
            noised_planes.append(plane_noise)
    s = len(clear_planes)

    p = np.random.permutation(s)
    c = np.stack(clear_planes)[p]
    n = np.stack(noised_planes)[p]

    np.save(os.path.join(dir_name,"clear_planes", 'train'), c[:int(s*0.6)])
    np.save(os.path.join(dir_name,"clear_planes", 'val'), c[int(s*0.6):int(s*0.8)])
    np.save(os.path.join(dir_name,"clear_planes", 'test'), c[int(s*0.8):])

    np.save(os.path.join(dir_name,"noised_planes", 'train'), n[:int(s*0.6)])
    np.save(os.path.join(dir_name,"noised_planes", 'val'), n[int(s*0.6):int(s*0.8)])
    np.save(os.path.join(dir_name,"noised_planes", 'test'), n[int(s*0.8):])

def crop_planes_and_dump(dir_name):
    for ss in ['train.npy', 'val.npy', 'test.npy']:
        clear_crops = []
        noised_crops = []
        c_name = os.path.join(dir_name,"clear_planes", ss)
        n_name = os.path.join(dir_name,"noised_planes", ss)

        for clear_plane, noised_plane in zip(np.load(c_name), np.load(n_name)):
            c = clear_plane[clear_plane!=np.inf].reshape([-1, clear_plane.shape[1]])
            n = noised_plane[noised_plane!=np.inf].reshape([-1, noised_plane.shape[1]])

            for clear_crop, noised_crop in get_crop(c, n, 500, (32,32)):
                clear_crops.append(clear_crop)
                noised_crops.append(noised_crop)
        np.save(os.path.join(dir_name,"clear_crops", ss), np.stack(clear_crops))
        np.save(os.path.join(dir_name,"noised_crops", ss), np.stack(noised_crops))


def main(source, dir_name):
    for i in ['clear_planes', 'clear_crops', 'noised_planes', 'noised_crops']:
        if not os.path.isdir(os.path.join(dir_name,i)):
            os.mkdir(os.path.join(dir_name,i))

    get_planes_and_dump(source, dir_name)
    crop_planes_and_dump(dir_name)

    for ss in ['train.npy', 'val.npy', 'test.npy']:
        print(ss, np.load(os.path.join(dir_name, 'clear_crops', ss)).shape)
        print(ss, np.load(os.path.join(dir_name, 'noised_crops', ss)).shape)

    

if __name__ == '__main__':
    args = vars(parser.parse_args())
    start = time.time()
    main(**args)
    print('Program done in %f'%(time.time()-start))
