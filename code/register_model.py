from mlflow.tracking import MlflowClient
import os,sys
sys.path.append('code')
from utils import mlflow_setup
mlflow_setup()
import mlflow
import yaml

def register_model(name="DVC_Mlflow",registry="DVC_Mlflow",experiment='mydemopipeline'):

    client = MlflowClient()
    
    EXPERIMENT_NAME = name
    MODEL_REGISTRY = registry
    EXPERIMENT_ID = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
    # client.create_registered_model(MODEL_REGISTRY) #mlflow.register_model instead

    # remote repository
    command = 'git remote get-url origin'
    output = os.popen(command).read().strip()
    http_url = output.replace('git@', 'https://').replace('.git', '')
      
    risk='This experiment was created to demonstrate to other users (e.g. model developers, data scientists, etc.) how can they develop, track, report and upload the final version of an automated pipeline. Under no circumstances it may be used for clinical decisions since the test train process is performed on a very small number of patients, the model is selected by the cross validation step and the training process has not been finalized thus creating a poor inference model.'

    # Create empty model registry




    runs = mlflow.search_runs(experiment_ids=EXPERIMENT_ID)
    exp_names = runs[runs['tags.mlflow.parentRunId'].isna()]
    pipeid = exp_names[exp_names['tags.mlflow.runName']==experiment].run_id.iloc[0]
    
    search = runs[runs['tags.mlflow.parentRunId'] == pipeid].run_id.tolist()
    search.append(pipeid)
    
    exp_results = runs[runs['tags.mlflow.parentRunId'].isin(search)].copy()
    tempdf = exp_results[exp_results['tags.mlflow.runName']=='Train'].filter(regex='param').dropna(axis=1).copy()
    art_path = exp_results[exp_results['tags.mlflow.runName']=='Train']['tags.Storage'].iloc[0]
    registered_run = exp_results[exp_results.artifact_uri == art_path.split('/model')[0]].run_id.iloc[0]
    metric=exp_results[exp_results['tags.mlflow.runName']=='Train'].filter(regex='metrics').dropna(axis=1).max(axis=1).iloc[0]

    
    tag_names = ['Name','Learning Problem','Learning Method','Storage','framework']
    # Create metadata dictionary 
    tags_dict = {
        'Experiment_name':EXPERIMENT_NAME,
        'Experiment_ID':EXPERIMENT_ID,
        'Experiment_run_id':pipeid,
        'Git repo':http_url,
        'Metric score':metric,
        'Risk Assessment':risk
    }

    tags_dict.update({key.split('.')[-1]:value.iloc[0] for key,value in tempdf.items()})

    tempdf = exp_results[exp_results['tags.mlflow.runName']=='Train'].filter(regex='tags').dropna(axis=1).copy()
    tags_dict.update({key.split('.')[-1]:value.iloc[0] for key,value in tempdf.items() 
                      if key.split('.')[-1] in tag_names})

    # result = client.create_model_version(
    #     name=MODEL_REGISTRY,
    #     source= art_path,
    #     run_id=registered_run,
    #     tags=tags_dict,
    #     description='Unet for prostate segmentation in T2 MRI',
    # )

    mlflow.register_model('runs:/'+registered_run+'/model', registry,tags=tags_dict)
    
    client.transition_model_version_stage(name=EXPERIMENT_NAME, version=1, stage="Production")
    client.update_model_version(name=EXPERIMENT_NAME, version=1, description='Unet for prostate segmentation in T2 MRI')
    
    model = client.get_registered_model(MODEL_REGISTRY)
    NAME = model.name
    VERSION = model.latest_versions[0].version
    client.set_model_version_tag(NAME,VERSION,'Author','Harry')
    
    client.set_registered_model_tag(MODEL_REGISTRY,'Task','Segmentation')

if __name__ == '__main__':
    register_model()