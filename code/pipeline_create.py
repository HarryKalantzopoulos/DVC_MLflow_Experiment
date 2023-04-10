import json
import mlflow
from utils import mlflow_setup
import sys
import yaml
from mlflow.tracking import MlflowClient
import os,shutil

os.makedirs('.temp',exist_ok=True)
temp = ".temp/pipeline.txt"

# Hyperparameters
params = yaml.safe_load(open("params.yaml"))
mlflow_activate = params["mlflow"]["activate"]
pipeline_name = params["mlflow"]["name"]

with open('DS_VERSION.json','r') as f:
    ds_log = json.load(f)
    
ds_tags ={'_'.join(['dataset',key]):ds_log['dataset'][key] for key in ds_log['dataset']}


if mlflow_activate:
    mlflow_setup()

    RUN_NAME = pipeline_name
    EXPERIMENT_NAME = "DVC_Mlflow"
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    parent_run = mlflow.start_run(run_name=pipeline_name)
    parent_runid = parent_run.info.run_id
    mlflow.set_tags(ds_tags)
    mlflow.set_tag("main_run_id", parent_runid)
    mlflow.set_tag("pipeline", "Preprocess;Prepare;Train")
    ds_log['pipeline']={'run_id':parent_runid}
    mlflow.end_run()
else:
    ds_log.pop('pipeline',None)
    
with open('DS_VERSION.json','w') as f:
    ds_log = json.dump(ds_log,f,indent=4)

with open(temp, "w") as f:
    f.write("pipeline_name")
    f.close()