

# --dataset=celebA --denoise_train --plot_reconstruction --gpus=0 --db_path=/ssd_data/CelebA --use_gui
# --dataset=lsun --denoise_train --plot_reconstruction --gpus=1 --db_path=/data/data/lsun/bedroom --use_gui

import argparse
import logging
import os

parser = argparse.ArgumentParser()
parser.add_argument('--gpus', type=str, default='')
parser.add_argument('--dataset', type=str, default='celebA')
parser.add_argument('--netname', type=str, default='')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--db_path', type=str, default='')
parser.add_argument('--version', type=int, default=1, help='A version number for this model.')
parser.add_argument('--continue_training', type=bool, default=True, 
                    help='Used to indicate we should continue training an existing model.')
parser.add_argument('--denoise_train', dest='denoise_train', action='store_true',
                    help='Use denoise training by adding Gaussian/salt and pepper noise')
parser.add_argument('--plot_reconstruction', dest='plot_reconstruction', action='store_true',
                    help='Plot reconstruction')
parser.add_argument('--use_gui', dest='use_gui', action='store_true',
                    help='Display the results with a GUI window')
parser.add_argument('--vis_frequency', type=int, default=1000, # VIS_FREQ = 1 EPOCH
                    help='How many train batches before we perform visualization')
args = parser.parse_args()

import matplotlib
if not args.use_gui:
    matplotlib.use('Agg')
else:
    from matplotlib import pyplot as plt
    plt.ion()
    plt.show()

from dataset import *
from sequential_vae import SequentialVAE
from trainer import NoisyTrainer

if args.gpus is not '':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

# Set the network name if there wasn't any passed in
if args.netname is None or args.netname == "":
    args.netname = "sequential_vae_%s" % dataset.name

# Check if there's an error using versions + continue flags
error = False
error_msg = ""
logpath = "models/" + args.netname + "_v" + str(args.version)
if not os.path.isdir(logpath) and args.continue_training:
    error = True
    error_message = "Cannot continue training a model if the file doesn't exist."
elif os.path.isdir(logpath) and not args.continue_training
    error = True
    error_message = "Model already exists. Either use a new version number, or, set the --continue_training flag to True"

# Construct logger
logpath = "models/" + args.netname + "_v" + str(args.version)
if not os.path.isdir(logpath):
    os.makedirs(logpath)
logging.basicConfig(filename=logpath+"/log.log", level=logging.DEBUG)
LOG = logging.getLogger(args.netname)

# Log + print error and quit gracefully if there was one
if error:
    print(error_msg)
    LOG.error(error_msg)
    exit(-1)

# Create the dataset object
if args.dataset == 'mnist':
    dataset = MnistDataset()
elif args.dataset == 'lsun':
    dataset = LSUNDataset(db_path=args.db_path)
elif args.dataset == 'celebA':
    dataset = CelebADataset(db_path=args.db_path)
elif args.dataset == 'svhn':
    dataset = SVHNDataset(db_path=args.db_path)
else:
    LOG.error("Unknown dataset")
    exit(-1)

# Construct network and trainer, then let it fly
model = SequentialVAE(dataset, name=args.netname, batch_size=args.batch_size, logger=LOG)
trainer = NoisyTrainer(model, dataset, args, LOG)
trainer.train()

