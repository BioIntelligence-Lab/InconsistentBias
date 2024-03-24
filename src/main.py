import argparse

import tensorflow as tf
from train import *
from test import *
from analysis import *

parser = argparse.ArgumentParser()
parser.add_argument('-train_test_aim_1_pnm', action='store_true')
parser.add_argument('-train_test_aim_1_ptx', action='store_true')
# parser.add_argument('-model', default='densenet', choices=['densenet', 'resnet', 'inception'])
# parser.add_argument('-test_ds', default='rsna', choices=['rsna', 'mimic', 'cxpt'])
# parser.add_argument('-test', action='store_true')
# parser.add_argument('-analyze', action='store_true')

args = parser.parse_args()
# model = args.model
# test_ds = args.test_ds

if __name__=='__main__':
  gpus = tf.config.list_physical_devices('GPU')
  if gpus:
    try:
      # Currently, memory growth needs to be the same across GPUs
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
      # Memory growth must be set before GPUs have been initialized
      print(e)
      
  # Run experiment based on passed arguments 
    
  if args.train_test_aim_1_pnm:
    train_aim_1_pnm()
    test_aim_1_pnm()
    
  if args.train_test_aim_1_ptx:
    train_aim_1_ptx()
    test_aim_1_ptx() 
      
  # if args.test:
  #   print(model, test_ds)
    
  # if args.analyze:
  #   analyze_aim_1_pnm()
  #   analyze_aim_1_ptx()