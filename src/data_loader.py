import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
from config import configs
from PIL import Image
import os
import numpy as np
import cv2
import h5py
import random
import pdb

class EventData(Dataset):
    """
    args:
    data_folder_path:the path of data
    split:'train' or 'test'
    """
    def __init__(self, data_folder_path, split, dt=1, transform=None):
        self._data_folder_path = data_folder_path
        self._split = split
        self.args = configs()
        self.dt = dt
        self.transform = transform
        if self._split == 'train':
            self.args.env = self.args.trainenv
        elif self._split == 'test':
            self.args.env = self.args.testenv

        self.p = 2 #number of polarities - fixed at 2 (+,-)

        #Image shape (x=vertical, y=horizontal)
        self.x = 260
        self.y = 346

        d_set = h5py.File(os.path.join(self._data_folder_path, self.args.env, self.args.env + '_data.hdf5'), 'r')
        # self.image_raw_event_inds = np.float64(d_set['davis']['left']['image_raw_event_inds'])
        self.image_raw_ts = np.float64(d_set['davis']['left']['image_raw_ts'])
        # gray image re-size
        self.length = d_set['davis']['left']['image_raw'].shape[0]
        d_set = None

        if self._split == 'train':
            self.index_th = 100
        elif self._split == 'test':
            self.index_th = 20
        else:
            self.index_th = 0


    def __getitem__(self, index):
        #Valid range in dataset
        if index + self.index_th < self.length and index > self.index_th:
            #Init
            count = np.zeros((self.dt, self.p, self.x, self.y), dtype=np.int16)
            time = np.zeros((self.dt, self.p, self.x, self.y), dtype=np.float64)
            gray = np.zeros((2, self.x, self.y), dtype=np.uint8) #first and last gray images

            #LOAD DATA########################################################################### 

            #Load Count and Time images
            for k in range(int(self.dt)):
                count[k] = np.load(os.path.join(self._data_folder_path, self.args.env, self.args.sub_dir, 'count_data', str(int(index+k+1))+'.npy'))
                time[k] = np.load(os.path.join(self._data_folder_path, self.args.env, self.args.sub_dir, 'time_data', str(int(index+k+1))+'.npy'))

            
            count = torch.from_numpy(count.astype(np.int16))
            count_image = torch.sum(count, dim=0).type(torch.float32)

            time_tensor = torch.from_numpy(time)
            time_max= torch.max(time_tensor, dim=0)[0]
            ts_st = self.image_raw_ts[int(index)]
            ts_end = self.image_raw_ts[int(index + self.dt)]

            time_range = ts_end - ts_st
            time_image = (time_max - ts_st) / time_range
            time_image = time_image.type(torch.FloatTensor) #Convert to float32 - range(0,1)

 
            #Load Gray images
            gray[0] = np.uint8(np.load(os.path.join(self._data_folder_path, self.args.env, self.args.sub_dir, 'gray_data', str(int(index))+'.npy')))
            gray[1] = np.uint8(np.load(os.path.join(self._data_folder_path, self.args.env, self.args.sub_dir, 'gray_data', str(int(index + self.dt))+'.npy')))

            gray_image = torch.from_numpy(gray.astype(np.uint8))    

            #Image shapes
            #Count: [2, 260, 346]
            #Time: [2, 260, 346]
            #Gray: [2, 260, 346]

            tcount = torch.zeros(self.p, self.args.image_height, self.args.image_width)
            ttime = torch.zeros(self.p, self.args.image_height, self.args.image_width)
            tgray = torch.zeros(self.p, self.args.image_height, self.args.image_width)

            #APPLY TRANSFORMATIONS########################################################################### 
            if self.transform:
                seed = np.random.randint(2147483647) #choose a random seed and fix it for all transformations

                for p in range(self.p):
                    #count
                    random.seed(seed)
                    torch.manual_seed(seed)
                    tcount[p] = self.transform(count_image[p])

                    #time
                    random.seed(seed)
                    torch.manual_seed(seed)
                    ttime[p] = self.transform(time_image[p])

                if self._split == 'train':
                    #gray[0]
                    random.seed(seed)
                    torch.manual_seed(seed)
                    tgray[0] = self.transform(gray_image[0])
                    #gray[1]
                    random.seed(seed)
                    torch.manual_seed(seed)
                    tgray[1] = self.transform(gray_image[1])
            
                    #Check for empty frames
                    if torch.max(tcount)>0 and torch.max(ttime)>0 and torch.max(tgray)>0:
                        return torch.cat((tcount, ttime), dim=0), tgray
                    else:
                        dummy = torch.zeros(self.p, self.args.image_height, self.args.image_width)
                        return torch.cat((dummy, dummy), dim=0), dummy
                elif self._split == 'test':
                    ts_st_end = torch.tensor([self.image_raw_ts[index], self.image_raw_ts[index+self.dt]], dtype=torch.float64)
                    #Check for empty frames
                    if torch.max(tcount)>0 and torch.max(ttime)>0:
                        return torch.cat((tcount, ttime), dim=0), ts_st_end
                    else:
                        dummy = torch.zeros(self.p, self.args.image_height, self.args.image_width)
                        dummy_ts = torch.zeros(2)
                        return torch.cat((dummy, dummy), dim=0), dummy_ts
            else:
                tcount = count_image
                ttime = time_image
                
                if self._split == 'train':
                    tgray = gray_image
            
                    #Check for empty frames
                    if torch.max(tcount)>0 and torch.max(ttime)>0 and torch.max(tgray)>0:
                        return torch.cat((tcount, ttime), dim=0), tgray
                    else:
                        dummy = torch.zeros(self.p, self.args.image_height, self.args.image_width)
                        return torch.cat((dummy, dummy), dim=0), dummy
                elif self._split == 'test':
                    ts_st_end = torch.tensor([self.image_raw_ts[index], self.image_raw_ts[index+self.dt]], dtype=torch.float64)
                    #Check for empty frames
                    if torch.max(tcount)>0 and torch.max(ttime)>0:
                        return torch.cat((tcount, ttime), dim=0), ts_st_end
                    else:
                        dummy = torch.zeros(self.p, self.args.image_height, self.args.image_width)
                        dummy_ts = torch.zeros(2)
                        return torch.cat((dummy, dummy), dim=0), dummy_ts
        else:
            if self._split == 'train':
                dummy = torch.zeros(self.p, self.args.image_height, self.args.image_width)
                return torch.cat((dummy, dummy), dim=0), dummy
            elif self._split == 'test':
                dummy = torch.zeros(self.p, self.args.image_height, self.args.image_width)
                dummy_ts = torch.zeros(2)
                return torch.cat((dummy, dummy), dim=0), dummy_ts

    def __len__(self):
        return self.length


class VoxelData(Dataset):
    def __init__(self, data_folder_path, split, dt=1, transform=None):
        self._data_folder_path = data_folder_path
        self._split = split
        self.transform = transform
        self.dt = dt
        self._split = split
        self.args = configs()
        self.nbins = 10

        #Image shape (x=vertical, y=horizontal)
        self.x = 260
        self.y = 346
        self.p = 2

        if self._split == 'train':
            self.args.env = self.args.trainenv
        elif self._split == 'test':
            self.args.env = self.args.testenv

        d_set = h5py.File(os.path.join(self._data_folder_path, self.args.env, self.args.env + '_data.hdf5'), 'r')
        self.image_raw_event_inds = np.float64(d_set['davis']['left']['image_raw_event_inds'])
        self.image_raw_ts = np.float64(d_set['davis']['left']['image_raw_ts'])
        # gray image re-size
        self.length = d_set['davis']['left']['image_raw'].shape[0]
        d_set = None

        if self._split == 'train':
            self.index_th = 100
        elif self._split == 'test':
            self.index_th = 20
        else:
            self.index_th = 0

    def __getitem__(self, index):
        #Valid range in dataset
        if index + self.index_th < self.length and index > self.index_th:
            count= np.zeros((self.p, self.x, self.y, self.dt*self.nbins), dtype=np.int16)
            gray = np.zeros((2, self.x, self.y), dtype=np.uint8) #first and last gray images

            #Load events
            for k in range(int(self.dt)):
                im_onoff = np.load(os.path.join(self._data_folder_path, self.args.env, self.args.sub_dir, 'count_data', str(int(index+k+1))+'.npy'))
                count[0, :, :, k*self.nbins:(k+1)*self.nbins] = im_onoff[0]
                count[1, :, :, k*self.nbins:(k+1)*self.nbins] = im_onoff[1]
            count = torch.from_numpy(count.astype(np.int16))
            count = count.type(torch.float32)

            #Load gray
            gray[0] = np.uint8(np.load(os.path.join(self._data_folder_path, self.args.env, self.args.sub_dir, 'gray_data', str(int(index))+'.npy')))
            gray[1] = np.uint8(np.load(os.path.join(self._data_folder_path, self.args.env, self.args.sub_dir, 'gray_data', str(int(index + self.dt))+'.npy')))
            gray_image = torch.from_numpy(gray.astype(np.uint8)) 

            #Image shapes
            #Count: [2, 260, 346, 10*dt]
            #Gray: [2, 260, 346]

            tcount = torch.zeros(self.p, self.args.image_height, self.args.image_width, self.dt*self.nbins)
            tgray = torch.zeros(self.p, self.args.image_height, self.args.image_width)

            #APPLY TRANSFORMATIONS########################################################################### 
            if self.transform:
                seed = np.random.randint(2147483647) #choose a random seed and fix it for all transformations

                for k in range(int(self.nbins*self.dt)):
                    # positive polarity
                    random.seed(seed)
                    torch.manual_seed(seed)
                    scale_a = count[0, :, :, k].max()
                    tcount[0, :, :, k] = self.transform(count[0, :, :, k])
                    if torch.max(tcount[0, :, :, k]) > 0:
                        tcount[0, :, :, k] = scale_a * tcount[0, :, :, k] / torch.max(tcount[0, :, :, k])

                    # negative polarity
                    random.seed(seed)
                    torch.manual_seed(seed)
                    scale_b = count[1, :, :, k].max()
                    tcount[1, :, :, k] = self.transform(count[1, :, :, k])
                    if torch.max(tcount[1, :, :, k]) > 0:
                        tcount[1, :, :, k] = scale_b * tcount[1, :, :, k] / torch.max(tcount[1, :, :, k])

                if self._split == 'train':
                    #gray[0]
                    random.seed(seed)
                    torch.manual_seed(seed)
                    tgray[0] = self.transform(gray_image[0])

                    #gray[1]
                    random.seed(seed)
                    torch.manual_seed(seed)
                    tgray[1] = self.transform(gray_image[1])

                    #Check for empty frames
                    if torch.max(tcount)>0 and torch.max(tgray)>0:
                        scount = tcount.permute(0,3,1,2).reshape(-1, self.args.image_height, self.args.image_width)
                        return scount, tgray
                    else:
                        dummyc = torch.zeros(self.p*self.dt*self.nbins, self.args.image_height, self.args.image_width)
                        dummyg = torch.zeros(self.p, self.args.image_height, self.args.image_width)
                        return dummyc, dummyg
                elif self._split == 'test':
                    ts_st_end = torch.tensor([self.image_raw_ts[index], self.image_raw_ts[index+self.dt]], dtype=torch.float64)
                    #Check for empty frames
                    if torch.max(tcount)>0:
                        scount = tcount.permute(0,3,1,2).reshape(-1, self.args.image_height, self.args.image_width)
                        return scount, ts_st_end
                    else:
                        dummyc = torch.zeros(self.p*self.dt*self.nbins, self.args.image_height, self.args.image_width)
                        dummy_ts = torch.zeros(2)
                        return dummyc, dummy_ts
            else:
                tcount = count
                if self._split == 'train':
                    tgray = gray_image
            
                    #Check for empty frames
                    if torch.max(tcount)>0 and torch.max(tgray)>0:
                        scount = tcount.permute(0,3,1,2).reshape(-1, self.args.image_height, self.args.image_width)
                        return scount, tgray
                    else:
                        dummyc = torch.zeros(self.p*self.dt*self.nbins, self.args.image_height, self.args.image_width)
                        dummyg = torch.zeros(self.p, self.args.image_height, self.args.image_width)
                        return dummyc, dummyg

                elif self._split == 'test':
                    ts_st_end = torch.tensor([self.image_raw_ts[index], self.image_raw_ts[index+self.dt]], dtype=torch.float64)
                    #Check for empty frames
                    if torch.max(tcount)>0:
                        scount = tcount.permute(0,3,1,2).reshape(-1, self.args.image_height, self.args.image_width)
                        return scount, ts_st_end
                    else:
                        dummyc = torch.zeros(self.p*self.dt*self.nbins, self.args.image_height, self.args.image_width)
                        dummy_ts = torch.zeros(2)
                        return dummyc, dummy_ts

        else:
            if self._split == 'train':
                dummyc = torch.zeros(self.p*self.dt*self.nbins, self.args.image_height, self.args.image_width)
                dummyg = torch.zeros(self.p, self.args.image_height, self.args.image_width)
                return dummyc, dummyg
            elif self._split == 'test':
                dummyc = torch.zeros(self.p*self.dt*self.nbins, self.args.image_height, self.args.image_width)
                dummy_ts = torch.zeros(2)
                return dummyc, dummy_ts

    def __len__(self):
        return self.length

def drawImageTitle(img, title):
    cv2.putText(img,
                title,
                (60, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                thickness=2,
                bottomLeftOrigin=False)
    return img

if __name__ == "__main__":

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop((256, 256), scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor(),
    ])

    #Test Time Encoding
    # data = EventData(data_folder_path='/local/a/akosta/Datasets/MVSEC/', split='train', dt=4, transform=train_transform)
    # print(data.length)
    # EventDataLoader = torch.utils.data.DataLoader(dataset=data, batch_size=1, shuffle=False)
    # for idx, input in enumerate(EventDataLoader):
    #     cntp = input[0][0,0].numpy()
    #     cntn = input[0][0,1].numpy()
    #     timep = input[0][0,2].numpy()
    #     timen = input[0][0,3].numpy()

    #     grayst = input[1][0,0].numpy()
    #     grayend = input[1][0,1].numpy()

    #     grayst = drawImageTitle(grayst, str(grayst.max()))
    #     cntp = drawImageTitle(cntp, str(cntp.max()))
    #     timep = drawImageTitle(timep, str(timep.max()))

    #     # pdb.set_trace()

    #     cv2.namedWindow('cnt')
    #     cv2.namedWindow('time')
    #     cv2.namedWindow('gray')

    #     cv2.imshow('cnt', cntp)
    #     cv2.imshow('time', timep)
    #     cv2.imshow('gray', grayst)
    #     cv2.waitKey(1)

    #Test Voxel Encoding
    data = VoxelData(data_folder_path='/local/a/akosta/Datasets/MVSEC/', split='train', dt=1, transform=train_transform)
    print(data.length)
    EventDataLoader = torch.utils.data.DataLoader(dataset=data, batch_size=1, shuffle=False)
    for idx, input in enumerate(EventDataLoader):
        # print(input[0].shape)
        # cntp = input[0][0,0].numpy()
        # cntn = input[0][0,1].numpy()
        # timep = input[0][0,2].numpy()
        # timen = input[0][0,3].numpy()

        grayst = input[1][0,0].numpy()
        grayend = input[1][0,1].numpy()

        grayst = drawImageTitle(grayst, str(grayst.max()))
        # cntp = drawImageTitle(cntp, str(cntp.max()))
        # timep = drawImageTitle(timep, str(timep.max()))

        # # pdb.set_trace()

        # cv2.namedWindow('cnt')
        # cv2.namedWindow('time')
        cv2.namedWindow('gray')

        # cv2.imshow('cnt', cntp)
        # cv2.imshow('time', timep)
        cv2.imshow('gray', grayst)
        cv2.waitKey(1)
