import os
import pandas as pd
import json

import local
from dataset import Dataset

num_trials = 25

def train_aim_1_pnm():
  for trial in range(num_trials):
    # RSNA Pneumonia (radiologist annotated)
    ckpt_dir = f'pnm/trial_{trial}/baseline_rad/'
    train_ds = Dataset(
      pd.read_csv(f'splits/pnm/trial_{trial}/train.csv'),
      ['Pneumonia_RAD']
    )
    val_ds = Dataset(
      pd.read_csv(f'splits/pnm/trial_{trial}/train.csv'),
      ['Pneumonia_RAD']
    )
    local.train_baseline(
      train_ds,
      val_ds,
      ckpt_dir
    )
    # NIH Pneumonia (automated labeller)
    ckpt_dir = f'pnm/trial_{trial}/baseline_nih/'
    train_ds = Dataset(
      pd.read_csv(f'splits/pnm/trial_{trial}/train.csv'),
      ['Pneumonia_NIH']
    )
    val_ds = Dataset(
      pd.read_csv(f'splits/pnm/trial_{trial}/train.csv'),
      ['Pneumonia_NIH']
    )
    local.train_baseline(
      train_ds,
      val_ds,
      ckpt_dir
    )
    
def train_aim_1_ptx():
  for trial in range(num_trials):
    # SIIM Pneumothorax (radiologist annotated)
    ckpt_dir = f'ptx/trial_{trial}/baseline_rad/'
    train_ds = Dataset(
      pd.read_csv(f'splits/ptx/trial_{trial}/train.csv'),
      ['Pneumothorax_RAD']
    )
    val_ds = Dataset(
      pd.read_csv(f'splits/ptx/trial_{trial}/train.csv'),
      ['Pneumothorax_RAD']
    )
    local.train_baseline(
      train_ds,
      val_ds,
      ckpt_dir
    )
    # NIH Pneumothorax (automated labeller)
    ckpt_dir = f'ptx/trial_{trial}/baseline_nih/'
    train_ds = Dataset(
      pd.read_csv(f'splits/ptx/trial_{trial}/train.csv'),
      ['Pneumothorax_NIH']
    )
    val_ds = Dataset(
      pd.read_csv(f'splits/ptx/trial_{trial}/train.csv'),
      ['Pneumothorax_NIH']
    )
    local.train_baseline(
      train_ds,
      val_ds,
      ckpt_dir
    )    
    