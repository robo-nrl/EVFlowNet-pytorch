import argparse
import os
import torch
from datetime import datetime
import json

def print_args(args):
    #Print args
    print(' ' * 20 + 'OPTIONS:')
    for k, v in vars(args).items():
        print(' ' * 20 + k + ': ' + str(v)) 

def configs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root-dir",
                        help="Root directory to save encoded data",
                        default='/local/a/akosta/Datasets/MVSEC/')
    parser.add_argument("--trainenv",
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
                        default='outdoor_day2')
    parser.add_argument("--testenv",
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
                        default='indoor_flying1')
    parser.add_argument("--sub-dir",
                        help="Sub-directory specific to implementation",
                        choices=['spf', #Spike-FlowNet - count, gray
                                'evf'], #Ev-FlowNet - count, time, gray   
                        default='evf')
    
    parser.add_argument("--save-path",
                        help="Save results path",
                        default='./results')

    parser.add_argument("--arch",
                        help="Architecture to use",
                        choices=['evf', #EvFlowNet
                                'minievf',
                                'microevf',
                                'nanoevf',
                                'picoevf',
                                'spf', #SpikeFlowNet
                                'fire'], #FireFlowNet
                        default='evf')
    parser.add_argument("--encoding",
                        help="Data Encoding",
                        choices=['time', #EvFlowNet original
                                'voxel',], #EvFlowNet modified
                        default='time')
    parser.add_argument('--batch-size',
                        type=int,
                        help="Training batch size.",
                        default=8)
    parser.add_argument('--start-epoch',
                        type=int,
                        help="Epoch start number",
                        default=0)
    parser.add_argument('--epochs',
                        type=int,
                        help="Total number of epochs",
                        default=150)
    
    parser.add_argument('--initial-learning-rate', '--lr',
                        type=float,
                        help="Initial learning rate.",
                        default=3e-4)
    parser.add_argument('--learning-rate-decay',
                        type=float,
                        help='Rate at which the learning rate is decayed.',
                        default=0.75)#0.9
    parser.add_argument('--smoothness-weight',
                        type=float,
                        help='Weight for the smoothness term in the loss function.',
                        default=0.5)
    parser.add_argument('--image-height',
                        type=int,
                        help="Image height.",
                        default=256)
    parser.add_argument('--image-width',
                        type=int,
                        help="Image width.",
                        default=256)
    parser.add_argument('--no-batch-norm',
                        action='store_true',
                        help='If true, batch norm will not be performed at each layer',
                        default=False)
    parser.add_argument('--dt',
                        type=int,
                        help='Number of consecutive event-volumes between grayscale images to use',
                        default=1)
    
    # Args for testing only.
    parser.add_argument('--pretrained',
                        type=str,
                        help="Pretrained model path")
    parser.add_argument('--render',
                        action='store_true',
                        help='If true, the flow predictions will be visualized during testing.',
                        default=False)
    parser.add_argument('--save-test-output',
                        action='store_true',
                        help='If true, output flow will be saved to a npz file.',
                        default='')
    parser.add_argument('--evaluate-interval',
                        type=int,
                        help='Evaluate after every N epochs',
                        default=2)
    parser.add_argument('--evaluate',
                        action='store_true',
                        help='If true, only run test',
                        default='')
    parser.add_argument('--gpus', 
                        type=str, 
                        help='gpus (default: 0)',
                        default='0')
    
    
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES']= args.gpus
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.arch == 'evf':
            args.base_channels = 64
    elif args.arch == 'minievf':
            args.base_channels = 32
    elif args.arch == 'microevf':
            args.base_channels = 16
    elif args.arch == 'nanoevf':
            args.base_channels = 8
    elif args.arch == 'picoevf':
            args.base_channels = 4
    elif args.arch == 'spf':
            args.base_channels = 64
    elif args.arch == 'fire':
            args.base_channels = 32

    if args.encoding == 'time':
            args.in_channels = 4
    elif args.encoding == 'voxel':
            args.in_channels = 10*args.dt*2

    args.gt_path = os.path.join(args.root_dir, args.testenv, args.testenv + '_gt.hdf5')
    args.hdf5_path = os.path.join(args.root_dir, args.testenv, args.testenv + '_data.hdf5')

    timestamp = timestamp = datetime.now().strftime("%m-%d-%H:%M")

    args.save_path = os.path.join(args.save_path, 'dt'+str(args.dt), args.arch, timestamp)

    if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)



    return args