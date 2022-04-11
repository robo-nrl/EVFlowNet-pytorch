#!/usr/bin/env python
import os
import time
import cv2
import pdb

import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F


from config import *
from data_loader import EventData
from eval_utils import *
from vis_utils import *
from EVFlowNet import EVFlowNet

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

def test(args, EventDataLoader, model):
    if args.render:
        cv2.namedWindow('EV-FlowNet Results', cv2.WINDOW_NORMAL)

    if args.gt_path:
        print("Loading ground truth {}".format(args.gt_path))
        d_gt = h5py.File(args.gt_path, 'r')
        gt_flow_raw = np.float32(d_gt['davis']['left']['flow_dist'])
        gt_ts = np.float64(d_gt['davis']['left']['flow_dist_ts'])
        print('Ground truth size: ', gt_flow_raw.shape)
        d_gt = None

        U_gt_all = np.array(gt_flow_raw[:, 0, :, :])
        V_gt_all = np.array(gt_flow_raw[:, 1, :, :])
    else:
        raise Exception("You need to provide ground truth path!")

    d_set = h5py.File(args.hdf5_path, 'r')
    gray= d_set['davis']['left']['image_raw']
    # pdb.set_trace()

    model.eval()

    AEE_sum = 0.
    percent_AEE_sum = 0.
    AEE_list = []

    if args.save_test_output:
        output_flow_list = []
        gt_flow_list = []
        event_image_list = []

    max_flow_sum = 0
    min_flow_sum = 0
    iters = 0

    if 'outdoor' in args.testenv:
        start1 = 9200
        stop1 = 9601
        start2 = 10500
        stop2 = 10901
    else:
        start1 = 0
        stop1 = len(EventDataLoader)
        start2 = 0
        stop2 = len(EventDataLoader)
    
    print('Start Frame1: {}, Stop Frame1 = {}'.format(start1, stop1))
    print('Start Frame2: {}, Stop Frame2 = {}'.format(start2, stop2))

    base_time = time.time()

    for idx, (input, ts) in enumerate(EventDataLoader):
        if torch.sum(input) > 0:
            if (start1 < idx < stop1) or (start2 < idx < stop2):
                start_time = time.time()
                flow_dict = model(input.to(args.device))
                
                fp_time = time.time() - start_time
                
                out_temp = np.squeeze(flow_dict['flow3'].detach().cpu().numpy())
                pred_flow = np.zeros((args.image_height, args.image_width, 2))
                pred_flow[:, :, 0] = cv2.resize(np.array(out_temp[0, :, :]), (args.image_height, args.image_width), interpolation=cv2.INTER_LINEAR)
                pred_flow[:, :, 1] = cv2.resize(np.array(out_temp[1, :, :]), (args.image_height, args.image_width), interpolation=cv2.INTER_LINEAR)
                pred_flow = np.flip(pred_flow, 2) ##loss func was flipped

                max_flow_sum += np.max(pred_flow)
                min_flow_sum += np.min(pred_flow)
                
                count_image = torch.sum(input[:, :2, ...], dim=1).numpy()
                count_image = (count_image * 255 / count_image.max()).astype(np.uint8)
                count_image = np.squeeze(count_image)

                event_mask = count_image.copy()
                event_mask[event_mask > 0] = 1

                # pdb.set_trace()
                U_gt, V_gt = estimate_corresponding_gt_flow(U_gt_all, V_gt_all, gt_ts, np.array(ts[0,0]), np.array(ts[0, 1]))   
                gt_flow = np.stack((U_gt, V_gt), axis=2)

                if args.save_test_output:
                    output_flow_list.append(pred_flow)
                    event_image_list.append(count_image)
                    gt_flow_list.append(gt_flow)
                    
                image_size = pred_flow.shape
                full_size = gt_flow.shape
                xsize = full_size[1]
                ysize = full_size[0]
                xcrop = image_size[1]
                ycrop = image_size[0]
                xoff = (xsize - xcrop) // 2
                yoff = (ysize - ycrop) // 2
                
                gt_flow = gt_flow[yoff:-yoff, xoff:-xoff, :]       
            
                # Calculate flow error.
                AEE, percent_AEE, n_points = flow_error_dense(gt_flow, 
                                                            pred_flow, 
                                                            count_image,
                                                            'outdoor' in args.testenv)
                AEE_list.append(AEE)
                AEE_sum += AEE
                percent_AEE_sum += percent_AEE
                    
                iters += 1
                if idx % 1000 == 0 and idx>10:
                    print('-------------------------------------------------------')
                    print('Idx: {}, Iter: {}, run time: {:.3f}s, time_elapsed: {:.3f}s\n'
                        'Mean max flow: {:.2f}, mean min flow: {:.2f}'
                        .format(idx, iters, fp_time, time.time()-base_time,
                                max_flow_sum/iters, min_flow_sum/iters))
                    if args.gt_path:
                        print('Mean AEE: {:.2f}, mean %AEE: {:.2f}, # pts: {:.2f}'
                            .format(AEE_sum/iters,
                                    percent_AEE_sum/iters,
                                    n_points))

                # Prep outputs for nice visualization.
                if args.render:
                    pred_flow_rgb = flow_viz_np(pred_flow[..., 0], pred_flow[..., 1])
                    pred_flow_rgb = drawImageTitle(pred_flow_rgb, 'Predicted Flow')
                    # cv2.imshow('Predicted Flow', cv2.cvtColor(pred_flow_rgb, cv2.COLOR_BGR2RGB))
                    
                    time_image = np.squeeze(np.amax(input[:, 2:, ...].numpy(), axis=1))
                    time_image = (time_image * 255 / time_image.max()).astype(np.uint8)
                    time_image = np.tile(time_image[..., np.newaxis], [1, 1, 3])
                    time_image = drawImageTitle(time_image, 'Timestamp Image')
                    # cv2.imshow('Timestamp Image', cv2.cvtColor(time_image, cv2.COLOR_BGR2RGB))
                    
                    
                    count_image = np.tile(count_image[..., np.newaxis], [1, 1, 3])
                    count_image = drawImageTitle(count_image, 'Count Image')
                    # cv2.imshow('Count Image', cv2.cvtColor(count_image, cv2.COLOR_BGR2RGB))
                    
                    
                    gray_image = cv2.resize(gray[idx], (args.image_height, args.image_width), interpolation=cv2.INTER_LINEAR)
                    gray_image = np.tile(gray_image[..., np.newaxis], [1, 1, 3])
                    gray_image = drawImageTitle(gray_image, 'Grayscale Image')
                    # cv2.imshow('Gray Image', cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB))
                    
                    gt_flow_rgb = np.zeros(pred_flow_rgb.shape)
                    errors = np.zeros(pred_flow_rgb.shape)

                    gt_flow_rgb = drawImageTitle(gt_flow_rgb, 'GT Flow - No GT')
                    errors = drawImageTitle(errors, 'Flow Error - No GT')
                    
                    if args.gt_path:
                        errors = np.linalg.norm(gt_flow - pred_flow, axis=-1)
                        errors[np.isinf(errors)] = 0
                        errors[np.isnan(errors)] = 0
                        errors = (errors * 255. / errors.max()).astype(np.uint8)
                        errors = np.tile(errors[..., np.newaxis], [1, 1, 3])
                        errors[count_image == 0] = 0

                        if 'outdoor' in args.testenv:
                            errors[190:, :] = 0
                        
                        gt_flow_rgb = flow_viz_np(gt_flow[...,0], gt_flow[...,1])

                        gt_flow_rgb = drawImageTitle(gt_flow_rgb, 'GT Flow')
                        errors= drawImageTitle(errors, 'Flow Error')

                    # cv2.imshow('Errors Image', cv2.cvtColor(errors, cv2.COLOR_BGR2RGB))
                    # cv2.imshow('Ground Truth', cv2.cvtColor(gt_flow_rgb, cv2.COLOR_BGR2RGB))
                        
                    top_cat = np.concatenate([count_image, gray_image, pred_flow_rgb], axis=1)
                    bottom_cat = np.concatenate([time_image, errors, gt_flow_rgb], axis=1)
                    cat = np.concatenate([top_cat, bottom_cat], axis=0)
                    cat = cat.astype(np.uint8)
                    cv2.imshow('EV-FlowNet Results', cat)
                    cv2.waitKey(1)

                    # pdb.set_trace()
                    
    print('Testing done. ')
    if args.gt_path:
        print('Mean AEE {:02f}, mean %AEE {:02f}'
            .format(AEE_sum/iters, 
                    percent_AEE_sum/iters))
    if args.save_test_output:
        if args.gt_path:
            print('Saving data to {}_output_gt.npz'.format(args.test_sequence))
            np.savez('{}_output_gt.npz'.format(args.test_sequence),
                    output_flows=np.stack(output_flow_list, axis=0),
                    gt_flows=np.stack(gt_flow_list, axis=0),
                    event_images=np.stack(event_image_list, axis=0))
        else:
            print('Saving data to {}_output.npz'.format(args.test_sequence))
            np.savez('{}_output.npz'.format(args.test_sequence),
                    output_flows=np.stack(output_flow_list, axis=0),
                    event_images=np.stack(event_image_list, axis=0))

    return AEE_sum/iters


def main():        
    args = configs()

    if not args.pretrained:
        raise Exception("You need to pass `pretrained` model path")

    model_data = torch.load(args.pretrained)
    model = EVFlowNet(args).to(args.device)
    model.load_state_dict(model_data['state_dict'])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor(),
    ])

    EventDataset = EventData(data_folder_path=args.root_dir, split='test', dt=args.dt, transform=test_transform)
    EventDataLoader = torch.utils.data.DataLoader(dataset=EventDataset, batch_size=1, shuffle=False)

    EPE = test(args, EventDataLoader, model)


if __name__ == "__main__":
    main()
