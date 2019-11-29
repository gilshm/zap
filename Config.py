import os
from Logger import Logger
from Datasets import Datasets
from models.alexnet_cifar import alexnet_cifar100
from models.alexnet_imagenet import alexnet
from models.vgg_imagenet import vgg16
from models.resnet_imagenet import resnet18

basedir, _ = os.path.split(os.path.abspath(__file__))
basedir = os.path.join(basedir, 'data')

MODELS = {'alexnet_cifar100': alexnet_cifar100,
          'alexnet_imagenet': alexnet,
          'vgg16_imagenet': vgg16,
          'resnet18_imagenet': resnet18}

BATCH_SIZE = 128


# ------------------------------------------------
#                   Directories
# ------------------------------------------------
CHECKPOINT_DIR = os.path.join(basedir, 'checkpoint')
RESULTS_DIR = os.path.join(basedir, 'results')
DATASET_DIR = os.path.join(basedir, 'datasets')
DATASET_DIR_IMAGENET = '/mnt/p3700/gilsho/ilsvrc2012'


# ------------------------------------------------
#              Statistics Parameters
# ------------------------------------------------
# Used for the values histograms
STATS_VAL_HIST_MIN = -2
STATS_VAL_HIST_MAX = 2
STATS_VAL_HIST_STEP = 0.1
STATS_VAL_HIST_BINS = int((STATS_VAL_HIST_MAX - STATS_VAL_HIST_MIN) / STATS_VAL_HIST_STEP)

# Used for error to threshold table
STATS_ERR_TO_TH_MIN = -1.3
STATS_ERR_TO_TH_MAX = 1.3
STATS_ERR_TO_TH_STEP = 0.1
STATS_ERR_TO_TH_MAX = STATS_ERR_TO_TH_MAX + STATS_ERR_TO_TH_STEP
STATS_ERR_TO_TH_BINS = round((STATS_ERR_TO_TH_MAX - STATS_ERR_TO_TH_MIN) / STATS_ERR_TO_TH_STEP)

# Used for the mask values histogram
STATS_MASK_VAL_HIST_MIN = -1.3
STATS_MASK_VAL_HIST_MAX = 1.3
STATS_MASK_VAL_HIST_STEP = 0.1
STATS_MASK_VAL_HIST_MAX = STATS_MASK_VAL_HIST_MAX + STATS_MASK_VAL_HIST_STEP
STATS_MASK_VAL_HIST_BINS = round((STATS_MASK_VAL_HIST_MAX - STATS_MASK_VAL_HIST_MIN) / STATS_MASK_VAL_HIST_STEP)

# Used for ROC
STATS_ROC_MIN = -2
STATS_ROC_MAX = 2
STATS_ROC_STEP = 0.2
STATS_ROC_MAX = STATS_ROC_MAX + STATS_ROC_STEP
STATS_ROC_BINS = round((STATS_ROC_MAX - STATS_ROC_MIN) / STATS_ROC_STEP)


# ------------------------------------------------
#                 Checkpoint paths
# ------------------------------------------------
# Modify checkpoints paths according to your specific folders

# AlexNet + CIFAR-100
chkps_alexnet_cifar100 = \
    {6: RESULTS_DIR + '/alexnet-cifar100_mask-6',
     5: RESULTS_DIR + '/alexnet-cifar100_mask-5',
     4: RESULTS_DIR + '/alexnet-cifar100_mask-4',
     3: RESULTS_DIR + '/alexnet-cifar100_mask-3'}

# AlexNet + ImageNet
chkps_alexnet_imagenet = \
    {6: RESULTS_DIR + '/alexnet-imagenet_mask-6',
     5: RESULTS_DIR + '/alexnet-imagenet_mask-5',
     4: RESULTS_DIR + '/alexnet-imagenet_mask-4',
     3: RESULTS_DIR + '/alexnet-imagenet_mask-3'}

# VGG-16 + ImageNet
chkps_vgg16_imagenet = \
    {6: RESULTS_DIR + '/vgg16-imagenet_mask-6',
     5: RESULTS_DIR + '/vgg16-imagenet_mask-5',
     4: RESULTS_DIR + '/vgg16-imagenet_mask-4',
     3: RESULTS_DIR + '/vgg16-imagenet_mask-3'}

# ResNet18 + ImageNet
chkps_resnet18_imagenet = \
    {6: RESULTS_DIR + '/resnet18-imagenet_mask-6',
     5: RESULTS_DIR + '/resnet18-imagenet_mask-5',
     4: RESULTS_DIR + '/resnet18-imagenet_mask-4',
     3: RESULTS_DIR + '/resnet18-imagenet_mask-3'}


# ------------------------------------------------
#                Init and Defines
# ------------------------------------------------
LOG = Logger()

STATS_GENERAL = 0
STATS_MISPRED_VAL_HIST = 1
STATS_MASK_VAL_HIST = 2
STATS_ERR_TO_TH = 3
STATS_ROC = 4

# ------------------------------------------------
#                  Util Functions
# ------------------------------------------------
def get_chkps_path(arch, dataset):
    if arch == 'alexnet' and dataset == 'cifar100':
        return chkps_alexnet_cifar100
    elif arch == 'alexnet' and dataset == 'imagenet':
        return chkps_alexnet_imagenet
    elif arch == 'vgg16' and dataset == 'imagenet':
        return chkps_vgg16_imagenet
    elif arch == 'resnet18' and dataset == 'imagenet':
        return chkps_resnet18_imagenet
    else:
        raise NotImplementedError


def get_model_chkp(arch, dataset):
    if arch == 'alexnet' and dataset == 'cifar100':
        filename = '_alexnet_cifar100_epoch-89_top1-64.38.pth'
        return '{}/{}'.format(RESULTS_DIR, filename)
    # For ImageNet PyTorch pretrained models are used
    elif dataset == 'imagenet':
        return None
    else:
        raise NotImplementedError


def get_dataset(dataset):
    if dataset == 'cifar100':
        return Datasets.get('CIFAR100', DATASET_DIR)
    elif dataset == 'imagenet':
        return Datasets.get('ImageNet', DATASET_DIR_IMAGENET)
    else:
        raise NotImplementedError


def get_mac_ops(arch, dataset):
    # TODO: the MAC operations shouldn't be defined here, but should probably be part of the model itself
    if arch == 'alexnet' and dataset == 'imagenet':
        return [80621568, 112140288, 149520384, 99680256]
    elif arch == 'resnet18' and dataset == 'imagenet':
        return [115605504, 115605504, 115605504, 115605504, 57802752, 115605504, 115605504, 115605504, 57802752,
                115605504, 115605504, 115605504, 57802752, 115605504, 115605504, 115605504]
    elif arch == 'vgg16' and dataset == 'imagenet':
        return [86704128, 1849688064, 924844032, 1849688064, 924844032, 1849688064, 1849688064, 924844032,
                1849688064, 1849688064, 462422016, 462422016, 462422016]
    else:
        raise NotImplementedError
