import json
import mlflow
from utils import mlflow_setup
import sys
import yaml
from mlflow.tracking import MlflowClient
import os,shutil
import ast # read imports in python files
import pkg_resources # for py package version
from importlib import import_module # for py package version if the above fails
from platform import python_version

#%% Find packages and versions used in each stage
def open_script(pypath):
    """
    Args:
        pypath(str) path to python file
    Returns:
        modules(list): names of imported python packages
    """
    with open(pypath, 'r') as file:
        code = file.read()

    tree = ast.parse(code)

    modules = set()
    for node in ast.walk(tree):
        # If the node is an import statement
        if isinstance(node, ast.Import):
            for name in node.names:
                if name.name not in modules: modules.add(name.name)
        # If the node is a from-import statement
        elif isinstance(node, ast.ImportFrom):
            if node.module not in modules: modules.add(node.module)
    return modules

def get_package_version(package_name):
    """
    Returns the version of the specified package, or None if the package does not have.
    """
    try:
        return pkg_resources.get_distribution(package_name).version
    except pkg_resources.DistributionNotFound:
        try:
            return import_module(package_name).__version__ 
        except:
            return None
# Input stage name
name = sys.argv[1]
temp = ".temp/{}.txt".format(name)

python_file = yaml.safe_load(open('dvc.yaml'))['stages'][name]['cmd'].split(' ')[-1]

modules_list = open_script(python_file)
module_version_dict = {mod:get_package_version(mod) 
                       for mod in modules_list 
                       if get_package_version(mod) is not None}


module_version_dict['python'] = python_version()
if name == 'Train':
    model_mod = open_script('code/model.py')
    model_mod_ver_dict = {mod:get_package_version(mod) 
                       for mod in model_mod 
                       if get_package_version(mod) is not None}
    
os.makedirs('.temp',exist_ok=True)
# Hyperparameters
params = yaml.safe_load(open("params.yaml"))
mlflow_activate = params["mlflow"]["activate"]

with open('DS_VERSION.json','r') as f:
    ds_log = json.load(f)
    
dvc_log = yaml.safe_load(open("dvc.lock"))


deps_dict = dvc_log['stages'][name]['deps']
cmd = dvc_log['stages'][name]['cmd'].split(' ')[-1]
outs_dict = dvc_log['stages'][name]['outs']

deps = {'depcode_'+i['path'] if i['path']==cmd else 'depdata_'+i['path']:i['md5'] 
        for i in deps_dict 
        if '.temp' not in i['path']}

outs = {'out_'+i['path']:i['md5'] for i in outs_dict} 

ds_log[name].update(module_version_dict)
ds_log[name].update(deps)
ds_log[name].update(outs)


with open('DS_VERSION.json','w') as f:
    json.dump(ds_log,f,indent=4)
    
    
if mlflow_activate:
    mlflow_setup()
    client = MlflowClient()
    RUN_NAME = name
    EXPERIMENT_NAME = "DVC_Mlflow"
    runid = ds_log[name]['run_id']
    

    code_str = ','.join([f"{k.split('depcode_')[-1]}${v}" for k, v in deps.items() if k.startswith('depcode_')])
    data_str = ','.join([f"{k.split('depdata_')[-1]}${v}" for k, v in deps.items() if k.startswith('depdata_')])

    
    client.set_tag(run_id=runid, key='depcode', value=code_str)
    client.set_tag(run_id=runid, key='depdata', value=data_str)
    
    #Output
    out_str = ','.join([f"{k.split('out_')[-1]}${v}" for k, v in outs.items() if k.startswith('out_')])
    # out_dict = {'out': data_str}
    
    client.set_tag(run_id=runid, key='out', value=out_str)

    
    mod_ver_string = ','.join([f"{k}${v}" for k, v in module_version_dict.items()])
    client.set_tag(run_id=runid, key='framework', value=mod_ver_string)
    
    if name == 'Train':
        model_mod_ver_string = ','.join([f"{k}${v}" for k, v in model_mod_ver_dict.items()])
        client.set_tag(run_id=runid, key='model_framework', value=model_mod_ver_string)
    
with open(temp, "w") as f:
    f.write("OK")
    f.close()
