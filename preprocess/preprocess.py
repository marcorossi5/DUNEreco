import numpy as np
import os
import argparse

from preprocessing_utils import load_files, get_planes, get_crop


parser = argparse.ArgumentParser()
parser.add_argument("--source", "-s", default="*", type=str, help='Source .npy file name inside clear(noised)_files directory')
parser.add_argument("--dir_name", "-d", default="../datasets", type=str, help='Directory path to datasets')

def main(source, dir_name):
    for i in ['clear_planes', 'clear_crops', 'noised_planes', 'noised_crops']:
        if not os.path.isdir(os.path.join(dir_name,i)):
            os.mkdir(os.path.join(dir_name,i))

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

    
    for ss in ['train.npy', 'val.npy', 'test.npy']:
        clear_crops = []
        noised_crops = []
        c_name = os.path.join(dir_name,"clear_planes", ss)
        n_name = os.path.join(dir_name,"noised_planes", ss)

        for clear_plane, noised_plane in zip(np.load(c_name), np.load(n_name)):
            c = clear_plane[clear_plane!=np.inf].reshape([-1, clear_plane.shape[1]])
            n = noised_plane[noised_plane!=np.inf].reshape([-1, noised_plane.shape[1]])

            for clear_crop, noised_crop in get_crop(c, n, 5, (32,32)):
                clear_crops.append(clear_crop)
                noised_crops.append(noised_crop)
        np.save(os.path.join(dir_name,"clear_crops", ss), np.stack(clear_crops))
        np.save(os.path.join(dir_name,"noised_crops", ss), np.stack(noised_crops))
    
    print(np.load(os.path.join(dir_name,"clear_crops", "train.npy")).shape)
    print(np.load(os.path.join(dir_name,"noised_crops", "train.npy")).shape)

if __name__ == '__main__':
    args = vars(parser.parse_args())
    main(**args)

#aggiungere creazioni delle cartelle clear_crops e clear_events in datasets