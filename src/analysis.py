
import numpy as np
import pandas as pd
from sklearn import metrics
from tqdm.auto import tqdm

num_trials = 25

# Metrics

def __threshold(y_true, y_pred):
  # Youden's J Statistic threshold
  fprs, tprs, thresholds = metrics.roc_curve(y_true, y_pred)
  return thresholds[np.nanargmax(tprs - fprs)]

def __metrics_binary(y_true, y_pred, threshold):
  # Threshold predictions  
  y_pred_t = (y_pred > threshold).astype(int)
  try:  
    auroc = metrics.roc_auc_score(y_true, y_pred)
  except:
    auroc = np.nan
    
  tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred_t, labels=[0,1]).ravel()
  if tp + fn != 0:
    tpr = tp/(tp + fn)
    fnr = fn/(tp + fn)
  else:
    tpr = np.nan
    fnr = np.nan
  if tn + fp != 0:
    tnr = tn/(tn + fp)
    fpr = fp/(tn + fp)
  else:
    tnr = np.nan
    fpr = np.nan
  if tp + fp != 0:
    fdr = fp/(fp + tp)
    ppv = tp/(fp + tp)
  else:
    ppv = np.nan
  if fn + tn != 0:
    npv = tn/(fn + tn)
    fomr = fn/(fn + tn)
  else:
    npv = np.nan
    fomr = np.nan
  return auroc, tpr, fnr, tnr, fpr, ppv, npv, fomr, tn, fp, fn, tp

def analyze_aim_1_pnm():
  results = []
  for ds in ['rad', 'nih']:
    for trial in range(num_trials):
      y_true = pd.read_csv(f'splits/pnm/trial_{trial}/test.csv')
      y_pred = pd.read_csv(f'results/pnm/trial_{trial}/baseline_{ds}_pred.csv')
      
      threshold_1 = __threshold(y_true['Pneumonia_RAD'], y_pred['Pneumonia_pred'])
      auroc_1, tpr_1, fnr_1, tnr_1, fpr_1, ppv_1, npv_1, fomr_1, tn_1, fp_1, fn_1, tp_1 = __metrics_binary(y_true['Pneumonia_RAD'], y_pred['Pneumonia_pred'], threshold_1)
      
      threshold_2 = __threshold(y_true['Pneumonia_NIH'], y_pred['Pneumonia_pred'])
      auroc_2, tpr_2, fnr_2, tnr_2, fpr_2, ppv_2, npv_2, fomr_2, tn_2, fp_2, fn_2, tp_2 = __metrics_binary(y_true['Pneumonia_NIH'], y_pred['Pneumonia_pred'], threshold_2)
      
      results += [
        [ds, 'rad', trial, np.nan, np.nan, auroc_1, tpr_1, fnr_1, tnr_1, fpr_1, ppv_1, npv_1, fomr_1, tn_1, fp_1, fn_1, tp_1],
        [ds, 'nih', trial, np.nan, np.nan, auroc_2, tpr_2, fnr_2, tnr_2, fpr_2, ppv_2, npv_2, fomr_2, tn_2, fp_2, fn_2, tp_2]
      ]

      for dem_sex in ['M', 'F']:
        y_true_t = y_true[y_true['Sex'] == dem_sex]
        y_pred_t = y_pred[y_pred['path'].isin(y_true_t['path'])]
        
        auroc_1, tpr_1, fnr_1, tnr_1, fpr_1, ppv_1, npv_1, fomr_1, tn_1, fp_1, fn_1, tp_1 = __metrics_binary(y_true_t['Pneumonia_RAD'], y_pred_t['Pneumonia_pred'], threshold_1)
        auroc_2, tpr_2, fnr_2, tnr_2, fpr_2, ppv_2, npv_2, fomr_2, tn_2, fp_2, fn_2, tp_2 = __metrics_binary(y_true_t['Pneumonia_NIH'], y_pred_t['Pneumonia_pred'], threshold_2)
        
        results += [
          [ds, 'rad', trial, dem_sex, np.nan, auroc_1, tpr_1, fnr_1, tnr_1, fpr_1, ppv_1, npv_1, fomr_1, tn_1, fp_1, fn_1, tp_1],
          [ds, 'nih', trial, dem_sex, np.nan, auroc_2, tpr_2, fnr_2, tnr_2, fpr_2, ppv_2, npv_2, fomr_2, tn_2, fp_2, fn_2, tp_2]
        ]
      
      for dem_age in ['0-20', '20-40', '40-60', '60-80', '80+']:
        y_true_t = y_true[y_true['Age_group'] == dem_age]
        y_pred_t = y_pred[y_pred['path'].isin(y_true_t['path'])]
        
        auroc_1, tpr_1, fnr_1, tnr_1, fpr_1, ppv_1, npv_1, fomr_1, tn_1, fp_1, fn_1, tp_1 = __metrics_binary(y_true_t['Pneumonia_RAD'], y_pred_t['Pneumonia_pred'], threshold_1)
        auroc_2, tpr_2, fnr_2, tnr_2, fpr_2, ppv_2, npv_2, fomr_2, tn_2, fp_2, fn_2, tp_2 = __metrics_binary(y_true_t['Pneumonia_NIH'], y_pred_t['Pneumonia_pred'], threshold_2)
        
        results += [
          [ds, 'rad', trial, np.nan, dem_age, auroc_1, tpr_1, fnr_1, tnr_1, fpr_1, ppv_1, npv_1, fomr_1, tn_1, fp_1, fn_1, tp_1],
          [ds, 'nih', trial, np.nan, dem_age, auroc_2, tpr_2, fnr_2, tnr_2, fpr_2, ppv_2, npv_2, fomr_2, tn_2, fp_2, fn_2, tp_2]
        ]
          
      for dem_sex in ['M', 'F']:
        for dem_age in ['0-20', '20-40', '40-60', '60-80', '80+']:
          y_true_t = y_true[(y_true['Sex'] == dem_sex) & (y_true['Age_group'] == dem_age)]
          y_pred_t = y_pred[y_pred['path'].isin(y_true_t['path'])]
          
          auroc_1, tpr_1, fnr_1, tnr_1, fpr_1, ppv_1, npv_1, fomr_1, tn_1, fp_1, fn_1, tp_1 = __metrics_binary(y_true_t['Pneumonia_RAD'], y_pred_t['Pneumonia_pred'], threshold_1)
          auroc_2, tpr_2, fnr_2, tnr_2, fpr_2, ppv_2, npv_2, fomr_2, tn_2, fp_2, fn_2, tp_2 = __metrics_binary(y_true_t['Pneumonia_NIH'], y_pred_t['Pneumonia_pred'], threshold_2)
          
          results += [
            [ds, 'rad', trial, dem_sex, dem_age, auroc_1, tpr_1, fnr_1, tnr_1, fpr_1, ppv_1, npv_1, fomr_1, tn_1, fp_1, fn_1, tp_1],
            [ds, 'nih', trial, dem_sex, dem_age, auroc_2, tpr_2, fnr_2, tnr_2, fpr_2, ppv_2, npv_2, fomr_2, tn_2, fp_2, fn_2, tp_2]
          ]
  results = np.array(results)
  df = pd.DataFrame(results, columns=['train', 'test', 'trial', 'dem_sex', 'dem_age', 'auroc', 'tpr', 'fnr', 'tnr', 'fpr', 'ppv', 'npv', 'fomr', 'tn', 'fp', 'fn', 'tp'])
  df.to_csv('results/pnm_summary.csv', index=False)
  
def analyze_aim_1_ptx():
  results = []
  for ds in ['rad', 'nih']:
    for trial in range(num_trials):
      y_true = pd.read_csv(f'splits/ptx/trial_{trial}/test.csv')
      y_pred = pd.read_csv(f'results/ptx/trial_{trial}/baseline_{ds}_pred.csv')
      
      threshold_1 = __threshold(y_true['Pneumothorax_RAD'], y_pred['Pneumothorax_pred'])
      auroc_1, tpr_1, fnr_1, tnr_1, fpr_1, ppv_1, npv_1, fomr_1, tn_1, fp_1, fn_1, tp_1 = __metrics_binary(y_true['Pneumothorax_RAD'], y_pred['Pneumothorax_pred'], threshold_1)
      
      threshold_2 = __threshold(y_true['Pneumothorax_NIH'], y_pred['Pneumothorax_pred'])
      auroc_2, tpr_2, fnr_2, tnr_2, fpr_2, ppv_2, npv_2, fomr_2, tn_2, fp_2, fn_2, tp_2 = __metrics_binary(y_true['Pneumothorax_NIH'], y_pred['Pneumothorax_pred'], threshold_2)
      
      results += [
        [ds, 'rad', trial, np.nan, np.nan, auroc_1, tpr_1, fnr_1, tnr_1, fpr_1, ppv_1, npv_1, fomr_1, tn_1, fp_1, fn_1, tp_1],
        [ds, 'nih', trial, np.nan, np.nan, auroc_2, tpr_2, fnr_2, tnr_2, fpr_2, ppv_2, npv_2, fomr_2, tn_2, fp_2, fn_2, tp_2]
      ]

      for dem_sex in ['M', 'F']:
        y_true_t = y_true[y_true['Sex'] == dem_sex]
        y_pred_t = y_pred[y_pred['path'].isin(y_true_t['path'])]
        
        auroc_1, tpr_1, fnr_1, tnr_1, fpr_1, ppv_1, npv_1, fomr_1, tn_1, fp_1, fn_1, tp_1 = __metrics_binary(y_true_t['Pneumothorax_RAD'], y_pred_t['Pneumothorax_pred'], threshold_1)
        auroc_2, tpr_2, fnr_2, tnr_2, fpr_2, ppv_2, npv_2, fomr_2, tn_2, fp_2, fn_2, tp_2 = __metrics_binary(y_true_t['Pneumothorax_NIH'], y_pred_t['Pneumothorax_pred'], threshold_2)
        
        results += [
          [ds, 'rad', trial, dem_sex, np.nan, auroc_1, tpr_1, fnr_1, tnr_1, fpr_1, ppv_1, npv_1, fomr_1, tn_1, fp_1, fn_1, tp_1],
          [ds, 'nih', trial, dem_sex, np.nan, auroc_2, tpr_2, fnr_2, tnr_2, fpr_2, ppv_2, npv_2, fomr_2, tn_2, fp_2, fn_2, tp_2]
        ]
      
      for dem_age in ['0-20', '20-40', '40-60', '60-80', '80+']:
        y_true_t = y_true[y_true['Age_group'] == dem_age]
        y_pred_t = y_pred[y_pred['path'].isin(y_true_t['path'])]
        
        auroc_1, tpr_1, fnr_1, tnr_1, fpr_1, ppv_1, npv_1, fomr_1, tn_1, fp_1, fn_1, tp_1 = __metrics_binary(y_true_t['Pneumothorax_RAD'], y_pred_t['Pneumothorax_pred'], threshold_1)
        auroc_2, tpr_2, fnr_2, tnr_2, fpr_2, ppv_2, npv_2, fomr_2, tn_2, fp_2, fn_2, tp_2 = __metrics_binary(y_true_t['Pneumothorax_NIH'], y_pred_t['Pneumothorax_pred'], threshold_2)
        
        results += [
          [ds, 'rad', trial, np.nan, dem_age, auroc_1, tpr_1, fnr_1, tnr_1, fpr_1, ppv_1, npv_1, fomr_1, tn_1, fp_1, fn_1, tp_1],
          [ds, 'nih', trial, np.nan, dem_age, auroc_2, tpr_2, fnr_2, tnr_2, fpr_2, ppv_2, npv_2, fomr_2, tn_2, fp_2, fn_2, tp_2]
        ]
          
      for dem_sex in ['M', 'F']:
        for dem_age in ['0-20', '20-40', '40-60', '60-80', '80+']:
          y_true_t = y_true[(y_true['Sex'] == dem_sex) & (y_true['Age_group'] == dem_age)]
          y_pred_t = y_pred[y_pred['path'].isin(y_true_t['path'])]
          
          auroc_1, tpr_1, fnr_1, tnr_1, fpr_1, ppv_1, npv_1, fomr_1, tn_1, fp_1, fn_1, tp_1 = __metrics_binary(y_true_t['Pneumothorax_RAD'], y_pred_t['Pneumothorax_pred'], threshold_1)
          auroc_2, tpr_2, fnr_2, tnr_2, fpr_2, ppv_2, npv_2, fomr_2, tn_2, fp_2, fn_2, tp_2 = __metrics_binary(y_true_t['Pneumothorax_NIH'], y_pred_t['Pneumothorax_pred'], threshold_2)
          
          results += [
            [ds, 'rad', trial, dem_sex, dem_age, auroc_1, tpr_1, fnr_1, tnr_1, fpr_1, ppv_1, npv_1, fomr_1, tn_1, fp_1, fn_1, tp_1],
            [ds, 'nih', trial, dem_sex, dem_age, auroc_2, tpr_2, fnr_2, tnr_2, fpr_2, ppv_2, npv_2, fomr_2, tn_2, fp_2, fn_2, tp_2]
          ]
  results = np.array(results)
  df = pd.DataFrame(results, columns=['train', 'test', 'trial', 'dem_sex', 'dem_age', 'auroc', 'tpr', 'fnr', 'tnr', 'fpr', 'ppv', 'npv', 'fomr', 'tn', 'fp', 'fn', 'tp'])
  df.to_csv('results/ptx_summary.csv', index=False)
  