import numpy as np
import json

configure_names = ['init_test_mot17', 'init_train_mot17',
                   'init_train_mot15', 'init_test_mot15', 'init_test_mot15_train_dataset',
                   'init_train_ua', 'init_test_ua']

current_select_configure = 'init_train_mot17'

config = {
    'mot_root': r'./dataset/MOT17',
    'save_folder': './weights/MOT17/weights0326-I50k-M80-G30',
    'log_folder': './weights/MOT17/log0326-I50k-M80-G30',
    'base_net_folder': './weights/MOT17/vgg16_reducedfc.pth',
    'resume': None,
    'start_iter': 55050,
    'cuda': True,
    'batch_size': 8,
    'num_workers': 16,
    'iterations': 85050,
    'learning_rate': 5e-3,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'false_constant': 10,
    'type': 'train',  # choose from ('test', 'train')
    'dataset_type': 'train',  # choose from ('test', 'train')
    'detector': 'FRCNN',  # choose from ('DPM', 'FRCNN', 'SDP')
    'max_object': 80,  # N
    'max_gap_frame': 40,  # not the hard gap
    'min_gap_frame': 0,  # not the hard gap
    'sst_dim': 900,
    'min_visibility': 0.3,
    'mean_pixel': (104, 117, 123),
    'max_expand': 1.2,
    'lower_contrast': 0.7,
    'upper_constrast': 1.5,
    'lower_saturation': 0.7,
    'upper_saturation': 1.5,
    'alpha_valid': 0.8,
    'crop_size': 112,
    'crop_delta': 1.5,
    'tensorboard': True
}

all_functions = []

'''
test mot train dataset
'''


def init_test_mot17():
    config['resume'] = './weights/sst300_0712_83000.pth'
    config['mot_root'] = './dataset/MOT17'
    config['save_folder'] = './seagate/weights0326-I50k-M80-G30'
    config['log_folder'] = './seagate/logs/1008-age-node'
    config['batch_size'] = 1
    config['write_file'] = True
    config['tensorboard'] = True
    config['save_combine'] = False
    config[
        'type'] = 'test'  # can be 'test' or 'train'. 'test' represents 'test dataset'. while 'train\ represents 'train dataset'


all_functions += [init_test_mot17]


def init_train_mot17():
    config['epoch_size'] = 664
    config['mot_root'] = 'D:/tracking_system/SST-master/dataset/MOT17'
    config['log_folder'] = './weights/MOT17/1031-E120-M80-G30-log'
    config['save_folder'] = './weights/MOT17/1031-E120-M80-G30-weights'
    config['save_images_folder'] = './weights/MOT17/1031-E120-M80-G30-images'
    config['type'] = 'train'
    config['resume'] = None  # None means training from sketch.
    config['detector'] = 'DPM'
    config['start_iter'] = 0
    config['iteration_epoch_num'] = 120
    config['iterations'] = config['start_iter'] + config['epoch_size'] * config['iteration_epoch_num'] + 50
    # config['batch_size'] = 4
    config['batch_size'] = 1
    config['learning_rate'] = 1e-2
    config['learning_rate_decay_by_epoch'] = (50, 80, 100, 110)
    config['save_weight_every_epoch_num'] = 5
    config['min_gap_frame'] = 0  # randomly select pair frames with the [min_gap_frame, max_gap_frame]
    config['max_gap_frame'] = 30
    config['false_constant'] = 1
    config['num_workers'] = 1
    config['cuda'] = True
    config['max_object'] = 60
    config['min_visibility'] = 0.3


all_functions += [init_train_mot17]


def init_train_mot15():
    config['epoch_size'] = 664
    config['mot_root'] = '/media/ssm/seagate/dataset/MOT15/2DMOT2015'
    config['base_net_folder'] = '/home/ssm/ssj/weights/MOT17/vgg16_reducedfc.pth'
    config['log_folder'] = '/home/ssm/ssj/weights/MOT15/1004-E120-M80-G30-log'
    config['save_folder'] = '/home/ssm/ssj/weights/MOT15/1004-E120-M80-G30-weights'
    config['save_images_folder'] = '/home/ssm/ssj/weights/MOT15/1004-E120-M80-G30-images'
    config['type'] = 'train'
    config['dataset_type'] = 'train'
    config['resume'] = None
    config['video_name_list'] = ['ADL-Rundle-6', 'ADL-Rundle-8', 'ETH-Bahnhof', 'ETH-Pedcross2', 'ETH-Sunnyday',
                                 'KITTI-13', 'KITTI-17', 'PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte', 'Venice-2']
    config['start_iter'] = 0
    config['iteration_epoch_num'] = 120
    config['iterations'] = config['start_iter'] + config['epoch_size'] * config['iteration_epoch_num'] + 50
    config['batch_size'] = 8
    config['learning_rate'] = 1e-2
    config['learning_rate_decay_by_epoch'] = (50, 80, 100, 110)
    config['save_weight_every_epoch_num'] = 5
    config['min_gap_frame'] = 0
    config['max_gap_frame'] = 30
    config['false_constant'] = 10
    config['num_workers'] = 16
    config['cuda'] = True
    config['max_object'] = 80
    config['min_visibility'] = 0.3
    config['final_net']['900'] = [int(config['final_net']['900'][0]), 1]


all_functions += [init_train_mot15]


def init_test_mot15():
    config['resume'] = '/media/ssm/seagate/weights/MOT17/0601-E120-M80-G30-weights/sst300_0712_83000.pth'
    config['mot_root'] = '/media/ssm/seagate/dataset/MOT15/2DMOT2015'
    config['log_folder'] = '/media/ssm/seagate/logs/1005-mot15-test-5'
    config['batch_size'] = 1
    config['write_file'] = True
    config['tensorboard'] = True
    config['save_combine'] = False
    config['type'] = 'test'
    config['dataset_type'] = 'test'
    config['video_name_list'] = ['ADL-Rundle-1', 'ADL-Rundle-3', 'AVG-TownCentre', 'ETH-Crossing', 'ETH-Jelmoli',
                                 'ETH-Linthescher', 'KITTI-16', 'KITTI-19', 'PETS09-S2L2', 'TUD-Crossing', 'Venice-1']


all_functions += [init_test_mot15]


def init_test_mot15_train_dataset():
    config['resume'] = '/media/ssm/seagate/weights/MOT17/0601-E120-M80-G30-weights/sst300_0712_83000.pth'
    config['mot_root'] = '/media/ssm/seagate/dataset/MOT15/2DMOT2015'
    config['log_folder'] = '/media/ssm/seagate/logs/1005-mot15-test-train-dataset-1'
    config['batch_size'] = 1
    config['write_file'] = True
    config['tensorboard'] = True
    config['save_combine'] = False
    config['type'] = 'train'
    config['dataset_type'] = 'train'
    config['video_name_list'] = ['ADL-Rundle-6', 'ADL-Rundle-8', 'ETH-Bahnhof', 'ETH-Pedcross2', 'ETH-Sunnyday',
                                 'KITTI-13',
                                 'KITTI-17', 'PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte', 'Venice-2']
    # config['video_name_list'] = ['KITTI-13']


all_functions += [init_test_mot15_train_dataset]


def init_train_ua():
    config['epoch_size'] = 10430

    config['base_net_folder'] = '/media/ssm/seagate/weights/UA-DETRAC/vgg16_reducedfc.pth'
    config['log_folder'] = '/media/ssm/seagate/weights/UA-DETRAC/0621-0728-E25-M80-G30-log'
    config['save_folder'] = '/media/ssm/seagate/weights/UA-DETRAC/0621-0728-E25-M80-G30-weight'
    config['save_images_folder'] = '/media/ssm/seagate/weights/UA-DETRAC/0621-0728-E25-M80-G30-images'
    config['ua_image_root'] = '/media/ssm/seagate/dataset/UA-DETRAC/Insight-MVT_Annotation_Train'
    config[
        'ua_detection_root'] = '/media/ssm/seagate/dataset/UA-DETRAC/gt'
    config[
        'ua_ignore_root'] = '/media/ssm/seagate/dataset/UA-DETRAC/DETRAC-MOT-toolkit/evaluation/igrs'
    config['resume'] = '/media/ssm/seagate/weights/UA-DETRAC/0621-0728-E25-M80-G30-weight/sst300_0712_62580.pth'
    config['start_iter'] = 62581
    config['iteration_epoch_num'] = 10
    config['iterations'] = config['start_iter'] + config['epoch_size'] * config['iteration_epoch_num'] + 50
    config['batch_size'] = 8
    config['learning_rate'] = 1e-3
    config['learning_rate_decay_by_epoch'] = (5, 7, 8, 9)
    config['save_weight_every_epoch_num'] = 1
    config['min_gap_frame'] = 0
    config['max_gap_frame'] = 15
    config['false_constant'] = 10
    config['num_workers'] = 16
    config['cuda'] = True
    config['max_object'] = 80


all_functions += [init_train_ua]


def init_test_ua():
    config['save_folder'] = '/media/ssm/seagate/weights/UA-DETRAC/1006-E25-M80-G30-TestSet-EB-1'
    config['ua_image_root'] = '/media/ssm/seagate/dataset/UA-DETRAC/Insight-MVT_Annotation_Test'
    config['ua_detection_root'] = '/media/ssm/seagate/dataset/UA-DETRAC/EB'
    config['ua_ignore_root'] = '/media/ssm/seagate/dataset/UA-DETRAC/DETRAC-MOT-toolkit/evaluation/igrs'
    config['resume'] = '/media/ssm/seagate/weights/UA-DETRAC/0621-0728-E25-M80-G30-weight/sst300_0712_114730.pth'
    config['detector_name'] = 'EB'
    config['batch_size'] = 1
    config['min_gap_frame'] = 0
    config['max_gap_frame'] = 30
    config['false_constant'] = 10
    config['cuda'] = True
    config['max_object'] = 80
    config['type'] = 'train'


all_functions += [init_test_ua]

for f in all_functions:
    if f.__name__ == current_select_configure:
        f()
        break

print('use configure: ', current_select_configure)
