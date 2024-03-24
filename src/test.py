import os
import pandas as pd
from tqdm.auto import tqdm
import json

import utils
from dataset import Dataset, union_labels

num_trials = 25

def test_aim_1_pnm():
  for trial in range(num_trials):
    for ds in ['rad', 'nih']:
      model_type = f'baseline_{ds}'
      ckpt_dir = f'pnm/trial_{trial}/{model_type}'
      os.makedirs('results/' + os.path.split(ckpt_dir)[0], exist_ok=True)
      # Load model
      model = utils.load_model(f'{ckpt_dir}/model.hdf5')
      # Set up test data
      test_ds = Dataset(
        pd.read_csv(f'splits/pnm/trial_{trial}/test.csv'),
        ['Pneumonia_RAD']
      )
      y_pred = model.predict(test_ds.get_dataset(shuffle=False))
      df = pd.DataFrame(pd.read_csv(f'splits/pnm/trial_{trial}/test.csv')['path'])
      df['Pneumonia_pred'] = y_pred
      df.to_csv(f'results/{ckpt_dir}_pred.csv', index=False)
      
def test_aim_1_ptx():
  for trial in range(num_trials):
    for ds in ['rad', 'nih']:
      model_type = f'baseline_{ds}'
      ckpt_dir = f'ptx/trial_{trial}/{model_type}'
      os.makedirs('results/' + os.path.split(ckpt_dir)[0], exist_ok=True)
      # Load model
      model = utils.load_model(f'{ckpt_dir}/model.hdf5')
      # Set up test data
      test_ds = Dataset(
        pd.read_csv(f'splits/ptx/trial_{trial}/test.csv'),
        ['Pneumothorax_RAD']
      )
      y_pred = model.predict(test_ds.get_dataset(shuffle=False))
      df = pd.DataFrame(pd.read_csv(f'splits/ptx/trial_{trial}/test.csv')['path'])
      df['Pneumothorax_pred'] = y_pred
      df.to_csv(f'results/{ckpt_dir}_pred.csv', index=False)
      