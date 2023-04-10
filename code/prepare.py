import os
from sklearn.model_selection import KFold
import pandas as pd
import yaml
import json
import mlflow
from utils import mlflow_setup,pipeline_keep_one_runid

#Hyper-parameters
params = yaml.safe_load(open("params.yaml"))

kfold = params["Prepare"]["kfold"]
mlflow_activate = params["mlflow"]["activate"]

# Load json log for pipeline
with open('DS_VERSION.json','r') as f:
    ds_log = json.load(f)
    
# ds_tags ={'_'.join(['dataset',key]):ds_log['dataset'][key] for key in ds_log['dataset']}

RUN_NAME = "Prepare"
EXPERIMENT_NAME = "DVC_Mlflow"
# Init Mlflow
if mlflow_activate:
    mlflow_setup()
    mlflow.set_experiment(EXPERIMENT_NAME)
    parent_runid = ds_log['pipeline']['run_id']
    mlflow.start_run(run_id=parent_runid)
    run = mlflow.start_run(run_name=RUN_NAME,nested=True)
    runid = run.info.run_id

output_fold = os.path.join("prepared", "kfold.json")

data = pd.read_csv(os.path.join("preprocess", "dataset.csv"))
ID = list(data['PatientID'])
X =  list(data['T2_path'])
y =  list(data['wg_path'])

kf = KFold(n_splits=kfold)
kf.get_n_splits(ID)

mydict={}
i=0
for train_idx, test_idx in kf.split(ID):
    ID_train = [ID[train] for train in train_idx]
    ID_test = [ID[test] for test in test_idx]
    fold="fold{}".format(i)

    mydict[fold]={'Train':ID_train,'Test':ID_test}
    i+=1

os.makedirs(os.path.join("prepared"), exist_ok=True)

with open(output_fold,'w') as f:
    json.dump(mydict,f,indent=4)
    
log_params = {'kfold':kfold}
ds_log[RUN_NAME] = log_params

#MLflow process
if mlflow_activate:
    mlflow.log_params(log_params)           
    # mlflow.set_tags(ds_tags)
    mlflow.set_tag("task", "kfold split")
    mlflow.set_tags({'previous_run_id':ds_log['Preprocess']['run_id']})
    #End run
    mlflow.end_run()
    # runid = check_mlflow_run_name_exist(RUN_NAME) #deletes previous runs
    pipeline_keep_one_runid(parent_runid,runid,RUN_NAME) #deletes previous runs
    ds_log[RUN_NAME]['run_id'] = runid
    mlflow.set_tag("Prepare_run_id", runid)
    mlflow.end_run()

    #Import Previous_step
    ds_log[RUN_NAME]['previous_run_id'] = ds_log['Preprocess']['run_id']

with open('DS_VERSION.json','w') as f:
    ds_log = json.dump(ds_log,f,indent=4)