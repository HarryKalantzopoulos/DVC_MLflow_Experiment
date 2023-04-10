import json
import yaml
import configparser #Reads config files
import os
import datetime

#Creates a fake output to keep the correct order when running
#DVC defines the correct execution sequence with inputs and outputs

os.makedirs('.temp',exist_ok=True)
temp = ".temp/read_dvc.txt"

#Read create and last modified date of the input dataset
input_folder = 'data'

created = os.path.getctime(input_folder)
created_datetime = f"{datetime.datetime.fromtimestamp(created)}"

modified = os.path.getmtime(input_folder)
modified_datetime = f"{datetime.datetime.fromtimestamp(modified)}"

mydict={}
mydict['dataset'] = {'created':created_datetime,'modified':modified_datetime}

#Type of data
file_list = [filename for _, _, filenames in os.walk(input_folder)
for filename in filenames]

ext_list=[]
for f in file_list:
    file,ext = os.path.splitext(f)
    fdtype = ext 
    while ext:
        file,ext = os.path.splitext(file)
        fdtype = ext + fdtype
    if fdtype not in ext_list:
        ext_list.append(fdtype)

mydict['dataset']['type'] = ",".join(ext_list)

#Reading info from data.dvc
ds_info = yaml.safe_load(open('data.dvc'))
ds_info = ds_info['outs'][0]
# key2store = ['md5','size','nfiles','path','desc','type']
key2store = ['md5','size','nfiles','path','desc']

mydict['dataset'].update({key:ds_info[key] for key in key2store})

for key in ds_info['meta']:
    mydict['dataset'][key] = ds_info['meta'][key]
    


config = configparser.ConfigParser()
config.read('.dvc/config')
mydict['dataset']['url'] = config['\'remote "storage"\'']['url']



with open('DS_VERSION.json','w') as f:
    json.dump (mydict,f,indent=4)
    
with open(temp, "w") as f:
    f.write("OK")
    f.close()