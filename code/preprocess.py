import SimpleITK as sitk
import numpy as np
import os
import yaml
import pandas as pd
import mlflow
from utils import loadITK,resample_spacing,write_better_json,cropRectangularProstate,crop_and_pad,norm8bit
from utils import pipeline_keep_one_runid,mlflow_setup
import json

#Hyper-parameters
params = yaml.safe_load(open("params.yaml"))
# Test data set split ratio
im_size = params["Preprocess"]['image_size']
p_sp = params["Preprocess"]["resample"] #tuple
crop = params["Preprocess"]["maskcrop"] #bool, False:centercrop
convert = params["Preprocess"]["8bit"] #bool
log_params=params["Preprocess"]

mlflow_activate = params["mlflow"]["activate"]

# Load json log for pipeline
with open('DS_VERSION.json','r') as f:
    ds_log = json.load(f)
    
# ds_tags ={'_'.join(['dataset',key]):ds_log['dataset'][key] for key in ds_log['dataset']}


RUN_NAME = "Preprocess"
EXPERIMENT_NAME = "DVC_Mlflow"

if mlflow_activate:
    # Init Mlflow
    mlflow_setup()
    mlflow.set_experiment(EXPERIMENT_NAME)
    parent_runid = ds_log['pipeline']['run_id']
    mlflow.start_run(run_id=parent_runid)
    run = mlflow.start_run(run_name=RUN_NAME,nested=True)
    runid = run.info.run_id

#Define output location
output_preprocess = os.path.join("preprocess")

output_csv = os.path.join(output_preprocess,"dataset.csv")
output_json = os.path.join(output_preprocess,"preprocess_log.json")

#Define existed patients,images and masks
T2 = sorted(os.listdir('data/images'))

names = [x.split('.mha')[0] for x in T2]
masks = [os.path.join(n+'.nii.gz') for n in names]

# masks = [os.path.join(n+'.nii.gz') for n in names]

log_dict={}
# log_params={}
count = 0

#Preprocess
preprocess_list = []
for idx,n in enumerate(names):

    img = loadITK(os.path.join('data','images',T2[idx]))
    msk = loadITK(os.path.join('data','masks',masks[idx]))


    log_dict[n]={
                "T2":{
                    "original_space":img.GetSpacing(),
                    "original_shape":img.GetSize()
                    },
                "wg":{
                    "original_space":msk.GetSpacing(),
                    "original_shape":msk.GetSize()                    
                    }
                }

    if p_sp and p_sp != None:
        if idx == 0:
            count += 1
            # log_params['preprocess{}.resample'.format(count)] = ",".join(map(str,list(p_sp)))
            preprocess_list.append(f'Image Resample')
        
        img,msk = resample_spacing(img,msk,out_spacing=p_sp)

        for obj in log_dict[n]:

            if obj == 'T2': x=img
            else: x=msk

            log_dict[n][obj].update({
                    "resampled_space":x.GetSpacing(),
                    "resampled_shape":x.GetSize()
            })

    else:
        for obj in log_dict[n]:
            log_dict[n][obj].update({
                        "resampled_space":False,
                        "resampled_shape":False
                })


    img_arr = sitk.GetArrayFromImage(img).astype("uint16")
    msk_arr = sitk.GetArrayFromImage(msk).astype("uint8")

    if im_size != None and isinstance(im_size,int):
        if idx==0:
            count += 1
            preprocess_list.append(f'Image Crop')
        
        
        crop_sz=int(im_size/2)
        # Custom json writer be sure to use apostrophes
        if crop: 
            center=cropRectangularProstate(msk_arr)
            typ = '\"whole_gland_crop\"'
            # if idx==0:
                # log_params['preprocess{}.crop_image_wg'.format(count)] = "{}x{}".format(im_size,im_size)
        else:
            x,y,_ = img.shape
            center = [x//2,y//2]
            typ = '\"center_crop\"'
            # if idx==0:
                # log_params['preprocess{}.crop_image_center'.format(count)] = "{}x{}".format(im_size,im_size)
                
        img_arr = crop_and_pad(img_arr,crop_sz,center)
        msk_arr = crop_and_pad(msk_arr,crop_sz,center)

        for obj in log_dict[n]:
            log_dict[n][obj]["crop"]=typ
            log_dict[n][obj]["crop_center"]=center
    
            if obj == 'T2': x=img_arr
            else: x=msk_arr
            log_dict[n][obj]["crop_shape"]=x.shape

    else:

        for obj in log_dict[n]:
            log_dict[n][obj]["crop"]=False
            log_dict[n][obj]["crop_center"]=False
            log_dict[n][obj]["crop_shape"]=False

    if convert:
        if idx == 0:
            count += 1
            # log_params['preprocess{}.8bitconversion'.format(count)] = True
            preprocess_list.append(f'Image Convert')
        img_arr = norm8bit(img_arr)
        conv ='true'

    
    #custom json writer is used True/False should change to true/false
    else: conv='false'

    log_dict[n]['T2']["8bit_convert"]=conv


    objects=["images","masks"]
    tosave = [img_arr,msk_arr]    
    asreference=[img,msk]

    for o in objects:
        os.makedirs(os.path.join(output_preprocess, o), exist_ok=True)

    for j,s,ref in zip(objects,tosave,asreference):
        temp = sitk.GetImageFromArray(s)
        temp.SetDirection(ref.GetDirection())
        temp.SetSpacing(ref.GetSpacing())
        temp.SetOrigin(ref.GetOrigin())
        sitk.WriteImage(temp, os.path.join(output_preprocess,j,names[idx]+".nii.gz"))

df =pd.DataFrame({
                'PatientID':names,
                # The name for both T2 and masks is ID.nii.gz == masks
                'T2_path':[os.path.join(output_preprocess,'images',file) for file in masks],
                'wg_path':[os.path.join(output_preprocess,'masks',file) for file in masks],
})

df.to_csv(output_csv,index=False)

write_better_json(log_dict,output_json)

ds_log[RUN_NAME] = log_params

if mlflow_activate:        
    mlflow.log_params(log_params)           
    mlflow.set_tag("task", "Data Transformation")
    mlflow.set_tag("description",",".join(preprocess_list))

    #End run and delete if other run_id exist
    mlflow.end_run()
    pipeline_keep_one_runid(parent_runid,runid,RUN_NAME) #deletes previous runs
    ds_log[RUN_NAME]['run_id'] = runid
    mlflow.set_tag("Preprocess_run_id", runid)
    #Import Previous_step

    mlflow.end_run()

# dvc_logs = yaml.safe_load(open("dvc.lock"))
# key = dvc_logs['stages']['dcm2nii']['deps'][1]['path']
# ds_log['DICOM2NIFTY'][key] = dvc_logs['stages']['dcm2nii']['deps'][1]['md5']

with open('DS_VERSION.json','w') as f:
    ds_log = json.dump(ds_log,f,indent=4)