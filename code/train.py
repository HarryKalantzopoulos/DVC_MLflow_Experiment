import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import yaml
import pandas as pd
import json
import time
import SimpleITK as sitk
import sys
sys.path.append('code')
from model import modelU
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import mlflow
from utils import loadITK,zscorestand,mlflow_setup
from utils import pipeline_keep_one_runid

from mlflow.keras import log_model
from mlflow.tensorflow import autolog
from tensorflow.keras import models
from mlflow.tracking import MlflowClient


#Hyper-parameters
params = yaml.safe_load(open("params.yaml"))

mlflow_activate = params["mlflow"]["activate"]

met = params['model']['metric']
nlab = params['model']['Number_labels']
plog = {key: params[key] for key in ['model','Train']}
params = params["Train"]
zscore = params['zscore']
batch = params["batch_size"]
e = params['epoch']

if plog['model']['architecture'] != 'ResUnet':
    del plog['model']['dilation']

# Create output folder
output_model=os.path.join('model')
os.makedirs(output_model,exist_ok=True)


# Load json log for pipeline
with open('DS_VERSION.json','r') as f:
    ds_log = json.load(f)

# ds_tags ={'_'.join(['dataset',key]):ds_log['dataset'][key] for key in ds_log['dataset']}


EXPERIMENT_NAME = "DVC_Mlflow"
RUN_NAME = "Train"
# Init Mlflow
if mlflow_activate:
    mlflow_setup()
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()
    EXPERIMENT_ID = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
    
    parent_runid = ds_log['pipeline']['run_id']
    mlflow.start_run(run_id=parent_runid)
    run = mlflow.start_run(run_name=RUN_NAME,nested=True)
    runid = run.info.run_id
    
    # mlflow.set_tags(ds_tags) #from DS_VERSION.json, meta-data for dataset in use
    [ mlflow.log_params(plog[k]) for k in plog ] #from paramls.yaml, hyper-parameters


# Inputs from kfold preparation step
dataset = pd.read_csv(os.path.join("preprocess", "dataset.csv"))

with open('prepared/kfold.json','r') as f:
    folds = json.load(f)

 #%% Modelcheckpoit and earlystoping
dur=[]
m=[]
fold=0
mdict={}
artifact_uri =[]

for kf in folds:

    model=modelU()

    ds = dataset[dataset['PatientID'].isin(folds[kf]['Train'])].copy()

    X=[loadITK(img,return_array=True).astype('float32') for img in ds['T2_path'] ]
    if zscore:
        X=[zscorestand(img).astype('float32') for img in X ]
    y=[loadITK(m,return_array=True).astype('float32') for m in ds['wg_path'] ]

    # Select slices that contains wg_mask
    X =np.array([im2 for im3 in X for im2 in im3])
    y =np.array([m2 for m3 in y for m2 in m3])

    X =np.array([img for img,m in zip(X,y) if np.max(m)==1])
    y =np.array([m for m in y if np.max(m)==1])

    X = np.expand_dims(X, axis=3)
    y = np.expand_dims(y, axis=3)

    if nlab == 2:
        y0 = np.where(y==0,1,0)
        y=np.concatenate([y0,y],axis=-1)

    save_weights_name= os.path.join(output_model,kf+'_model.h5')

    MC = ModelCheckpoint(
        save_weights_name, monitor=met,verbose=1, save_best_only=True,
        save_weights_only=False, mode='max', save_freq='epoch')
    
    # Number of children mlflow runs equal to number of kfolds
    if mlflow_activate:
        with mlflow.start_run(run_name='fold{}'.format(fold),nested=True) as fold_run:
            autolog()
            start_time = time.time()
            history=model.fit(X,y,batch_size=batch,
                    epochs=e,verbose=1,
                    callbacks=[MC])
            duration=(time.time() - start_time)/60
            artifact_uri.append(fold_run.info.artifact_uri)
            m.append(max(history.history[met]))
            dur.append(duration)
            # log_model(model, os.path.join(output_model,kf+'_model.h5'))
            
    else:
            start_time = time.time()
            history=model.fit(X,y,batch_size=batch,
                    epochs=e,verbose=1,
                    callbacks=[MC])
            duration=(time.time() - start_time)/60
            m.append(max(history.history[met]))
            dur.append(duration)

    mdict['fold{}'.format(fold)] = {met:m,'duration':dur}
    # mlflow.log_metrics({'Train_fold{}'.format(fold)+'_'+met:,m,'Train_fold{}'.format(fold)+'_duration':dur})
    fold+=1

with open(os.path.join(output_model,'folds.json'),'w') as f:
    json.dump(mdict,f,indent=4)


#Evaluation process
kfold =0
iou = []
metric_max = 0

for kf in folds:
    ds = dataset[dataset['PatientID'].isin(folds[kf]['Test'])].copy()

    X=[loadITK(img,return_array=True).astype('float32') for img in ds['T2_path'] ]
    if zscore:
        X=[zscorestand(img).astype('float32') for img in X ]
    y=[loadITK(m,return_array=True).astype('float32') for m in ds['wg_path'] ]

    # Select slices that contains wg_mask
    X =np.array([im2 for im3 in X for im2 in im3])
    y =np.array([m2 for m3 in y for m2 in m3])

    X =np.array([img for img,m in zip(X,y) if np.max(m)==1])
    y =np.array([m for m in y if np.max(m)==1])

    X = np.expand_dims(X, axis=3)
    y = np.expand_dims(y, axis=3)

    if nlab == 2:
        y0 = np.where(y==0,1,0)
        y=np.concatenate([y0,y],axis=-1)

    model = models.load_model('model/fold{}_model.h5'.format(kfold))
    pred=model.predict(X)
    metric = tf.keras.metrics.OneHotIoU(num_classes=2, target_class_ids=[1])
    metric.update_state(y,pred)
    iou.append(metric.result().numpy())
    
    if metric.result().numpy() > metric_max:
        metric_max = metric.result().numpy()
        keepfile = artifact_uri[kfold]+'/model/fold{}_model.h5'.format(kfold)
    kfold+=1
    
csv = pd.DataFrame(data={'fold':[i for i in range(kfold)],met:iou})
csv.to_csv('model/metric.csv',index=False)

e_metric = {"Eval_fold{}_IoU".format(i):m for i, m, in enumerate(iou)}

log_params = plog['model']
log_params.update(plog['Train'])
ds_log[RUN_NAME] = log_params

if mlflow_activate:
    mlflow.log_metrics(e_metric)
    mlflow.set_tags({"Task":"Model Train & Evaluation",
                     "Name":"Prostate Whole Gland Segmentation",
                     "Learning Problem": "Deep Learning",
                     "Learning Method": "Supervised",
                     "Storage": keepfile,
                    })

    ds_log[RUN_NAME]['previous_run_id'] = ds_log['Prepare']['run_id']
    mlflow.set_tags({'previous_run_id':ds_log['Prepare']['run_id']})
    mlflow.end_run()
    pipeline_keep_one_runid(parent_runid,runid,RUN_NAME) #deletes previous runs

    
    query = "tags.mlflow.parentRunId = '{}'".format(runid)
    child_ids = mlflow.search_runs(experiment_ids=[EXPERIMENT_ID], filter_string=query)

    ds_log[RUN_NAME]['run_id'] = runid
    
    mlflow.set_tag("Train_run_id", runid)
    

    for idx in range(len(child_ids)):
        temp = child_ids['tags.mlflow.runName'].iloc[idx]
        mlflow.set_tag("Train_"+temp+'_run_id', child_ids['run_id'].iloc[idx])
        ds_log[RUN_NAME]["Train_"+temp+'_run_id'] = child_ids['run_id'].iloc[idx]
    mlflow.end_run()
    
with open('DS_VERSION.json','w') as f:
    ds_log = json.dump(ds_log,f,indent=4)
