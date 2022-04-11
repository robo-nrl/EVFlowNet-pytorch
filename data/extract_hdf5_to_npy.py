#!/usr/bin/env python

import os
import argparse
import h5py
from tqdm import trange
from multiprocessing import Pool as ThreadPool
import numpy as np
import pdb

def encode_single(args, events=0, gray=0, image_raw_event_inds=0):
    img_gray = gray
    graylen = img_gray.shape[0]

    img_c = np.zeros((2, gray.shape[1], gray.shape[2]), dtype=np.int16)
    img_t = np.zeros((2, gray.shape[1], gray.shape[2]), dtype=np.float64)

    for i in trange(graylen):
        if i == 0:
            data_c = events[0:image_raw_event_inds[i], :]
        else:
            data_c = events[image_raw_event_inds[i-1]:image_raw_event_inds[i], :]

        if data_c.size > 0:
            img_c.fill(0)
            img_t.fill(0)

            for idx in range(data_c.shape[0]):
                y = data_c[idx, 1].astype(int)
                x = data_c[idx, 0].astype(int)
                t = data_c[idx, 2]

                if data_c[idx, 3].item() == -1:
                    img_c[1, y, x] += 1
                    img_t[1, y, x] = t
                elif data_c[idx, 3].item() == 1:
                    img_c[0, y, x] += 1
                    img_t[0, y, x] = t

            #No Normalization - Do in dataloader based on dt=1,2,3,4 ...
            img_time = img_t #np.array(img_t.shape, dtype=np.uint16)
            img_count = img_c

            np.save(os.path.join(args.count_dir, str(i)), img_count)
            np.save(os.path.join(args.gray_dir, str(i)), img_gray[i,:,:])
            np.save(os.path.join(args.time_dir, str(i)), img_time)

def preprocess(raw_events, gray, raw_event_inds):
    data_c, img_gray, indices = [], [], []
    
    for i in range(gray.shape[0]):
        if raw_event_inds[i-1] < 0:
            data_c.append(raw_events[0:raw_event_inds[i], :])
        else:
            data_c.append(raw_events[raw_event_inds[i-1]:raw_event_inds[i], :])

        img_gray.append(gray[i, :, :])
        indices.append(i)
    
    return data_c, img_gray, indices


def encode_mp(data):
    global args
    data_c, img_gray, i = data
    
    # print('Gray shape:', gray.shape)
    # print('Frame shape:', frame_data.shape)
    # print('Idx:', idx)
    
    img_c = np.zeros((2, img_gray.shape[0], img_gray.shape[1]), dtype=np.int16)
    img_t = np.zeros((2, img_gray.shape[0], img_gray.shape[1]), dtype=np.float64)
    
    if data_c.size > 0:
        img_c.fill(0)
        img_t.fill(0)

        for idx in range(data_c.shape[0]):
            y = data_c[idx, 1].astype(int)
            x = data_c[idx, 0].astype(int)
            t = data_c[idx, 2]

            if data_c[idx, 3].item() == -1:
                img_c[1, y, x] += 1
                img_t[1, y, x] = t
            elif data_c[idx, 3].item() == 1:
                img_c[0, y, x] += 1
                img_t[0, y, x] = t

        #No Normalization - Do in dataloader based on dt=1,2,3,4 ...
        img_time = img_t
        img_count = img_c

        np.save(os.path.join(args.count_dir, str(i)), img_count)
        np.save(os.path.join(args.gray_dir, str(i)), img_gray)
        np.save(os.path.join(args.time_dir, str(i)), img_time)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Extracts grayscale,event and timestamp images from a HDF5 file and "
                     "saves them as .npy files for training in Pytorch."))
    parser.add_argument("--root-dir",
                        help="Root directory to save encoded data",
                        required=True)
    parser.add_argument("--env",
                        help="Environment Name",
                        choices=['indoor_flying1', 
                                'indoor_flying2', 
                                'indoor_flying3',
                                'indoor_flying4',
                                'outdoor_day1',
                                'outdoor_day2',
                                'outdoor_night1',
                                'outdoor_night2',
                                'outdoor_night3'],
                        required=True)
    parser.add_argument("--sub-dir",
                        help="Sub-directory specific to implementation",
                        choices=['spf', #Spike-FlowNet - count, gray
                                'evf'], #Ev-FlowNet - count, gray, time    
                        required=True)
    parser.add_argument('--mp', 
                        help='Use Multiprocessing',
                        action="store_true")
    

    args = parser.parse_args()

    # Initialize Directory structure
    args.load_path = os.path.join(args.root_dir, args.env, args.env + '_data.hdf5')

    args.save_path = os.path.join(args.root_dir, args.env, args.sub_dir)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    args.count_dir = os.path.join(args.save_path, 'count_data')
    if not os.path.exists(args.count_dir):
        os.makedirs(args.count_dir)
      
    args.gray_dir = os.path.join(args.save_path, 'gray_data')
    if not os.path.exists(args.gray_dir):
        os.makedirs(args.gray_dir)

    args.time_dir = os.path.join(args.save_path, 'time_data')
    if not os.path.exists(args.time_dir):
        os.makedirs(args.time_dir)


    #Print args
    print(' ' * 20 + 'OPTIONS:')
    for k, v in vars(args).items():
        print(' ' * 20 + k + ': ' + str(v))

    
    #Load data from HDF5 file
    d_set = h5py.File(args.load_path, 'r')
    raw_data = d_set['davis']['left']['events'] # [t, y, x, p]
    image_raw_event_inds = d_set['davis']['left']['image_raw_event_inds']
    image_raw_ts = np.float64(d_set['davis']['left']['image_raw_ts'])
    gray_image = d_set['davis']['left']['image_raw']
    d_set = None
    print('Grayscale image shape: {}'.format(gray_image.shape))

    # pdb.set_trace()

    if not args.mp:
        # Without MP
        print('Saving encoded files using single process...')
        encode_single(args, events=raw_data, gray=gray_image, image_raw_event_inds=image_raw_event_inds)
        raw_data = None
    else:
        # With MP
        print('Preprocessing...')
        data_c, img_gray, indices = preprocess(raw_data, gray_image, image_raw_event_inds)
        raw_data = None
        print('Done!')
        
        print('Saving encoded files using mutiprocessing...')
        pool = ThreadPool()
        pool.map(encode_mp, zip(data_c, img_gray, indices))
        pool.close()
        pool.join()