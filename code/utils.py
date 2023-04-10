import SimpleITK as sitk
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
import os

def zscorestand(img):
    '''
    Standarize given image (array).
    '''
    img = img.clip(np.quantile(img,0.05),np.quantile(img,0.95))
    img = (img - img.mean()) / (img.std() + 1e-8)
    return img


def loadITK(file_path,return_array=False,orientation='LPS'):
    '''
    Load medical images as ITK images or numpy arrays.
    Args:
        file_path(str): path to image (.dcm,.nii.gz,.mha)
        return_array(bool): True returns numpy array instead of ITKimage
        orientation(str,default='LPS'): On which orientation you want to return the image
    Return:
        ITK_image or numpy array
    '''
    ITK_image=sitk.ReadImage(file_path)
    ITK_image=sitk.DICOMOrient(ITK_image, orientation)
    if return_array:
        return sitk.GetArrayFromImage(ITK_image)
    else:
        return ITK_image


def resample_spacing(
    image,
    mask,
    out_spacing = (2.0, 2.0, 2.0),
    out_size = None,
    pad_value = 0.):
    """
    Resample images to target resolution spacing
    Ref: SimpleITK
    """
    # get original spacing and size
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    # convert our z, y, x convention to SimpleITK's convention
    out_spacing = list(out_spacing)[::-1]

    if out_size is None:
        # calculate output size in voxels
        out_size = [
            int(np.round(
                size * (spacing_in / spacing_out)
            ))
            for size, spacing_in, spacing_out in zip(original_size, original_spacing, out_spacing)
        ]

    # determine pad value
    if pad_value is None:
        pad_value = image.GetPixelIDValue()

    # Resample image
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(list(out_spacing))
    resample.SetSize(out_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(pad_value)
    resample.SetInterpolator(sitk.sitkBSpline)
    image = resample.Execute(image)

    # Resample mask 
    # (masks' spacing are using image spacing as reference, despite if value exists or 1)
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(list(out_spacing))
    resample.SetSize(out_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(pad_value)
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    mask = resample.Execute(mask)
    
    return image,mask



def write_better_json(mydict,fname):
    '''
    Create a json of logging the preprocessing steps for every image in a more human readable format.
    '''
    with open(fname, "w") as outfile:
        outfile.write('{\n')
        patcheck = list(mydict)[-1]
        for pat in mydict:
            outfile.write("    \"{}\": {}\n".format(pat,'{'))
            objcheck = list(mydict[pat])[-1]
            for obj in mydict[pat]:
                outfile.write("       \"{}\": {}\n".format(obj,'{'))
                icheck = list(mydict[pat][obj])[-1]
                for i in mydict[pat][obj]:
                    t = mydict[pat][obj][i]
                    if isinstance(t,tuple): t = list(t)
                    
                    if i != icheck:
                        outfile.write("           \"{}\": {},\n".format(i,t))
                    else: #last element
                        outfile.write("           \"{}\": {}\n".format(i,t))
                if objcheck != obj:
                    outfile.write("        },\n")
                else:
                    outfile.write("        }\n")
            if patcheck != pat:
                outfile.write("    },\n")
            else:
                outfile.write("    }\n}")

def cropRectangularProstate(wg_mask):
    # location of wg slice with largest surface
    A_idx=np.argmax(np.sum(wg_mask,axis=(1,2)))
    [x,y]=np.nonzero(wg_mask[A_idx,:,:])
    x_center = (np.max(x)-np.min(x))//2+np.min(x)
    y_center = (np.max(y)-np.min(y))//2+np.min(y)
    return x_center,y_center

def crop_and_pad(image,crop,wg_centers):
    temp=image[:,wg_centers[0]-crop:wg_centers[0]+crop,wg_centers[1]-crop:wg_centers[1]+crop]
    if temp.shape[1]<2*crop:
        temp= np.pad(temp, [(0,0),(0,2*crop-temp.shape[1]), (0,0)], mode='constant')
    if temp.shape[2]<2*crop:
        temp= np.pad(temp, [(0,0),(0,0), (0,2*crop-temp.shape[2])], mode='constant')
    return temp


def norm8bit(mynumpy):

        mn = mynumpy.min()
        mx = mynumpy.max()
        mx -= mn
        if mx !=0:
            return (((mynumpy - mn)/mx) * 255).astype(np.uint8)
        else:
            raise ValueError("3D Image has no intensity range")

def mlflow_setup():
    os.environ["LOGNAME"] = "Harry"
    os.environ['GIT_PYTHON_REFRESH'] = "quiet"
    os.environ['AWS_ACCESS_KEY_ID'] = 'admin'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'admin1234'
    os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000/'
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000/'
    
def pipeline_keep_one_runid(parent_runid,id_to_retain,stage,EXPERIMENT_NAME = "DVC_Mlflow"):
    '''
    If a nested stage already exists, it keeps only the one which is currently active.
    Args:
        parent_runid(str): id of the pipeline
        id_to_retain(str): new run id of the reruned stage
        stage(str): stage run name
        EXPERIMENT_NAME(str): Name of your experiment, default DVC_Mlflow
    Returns:
        None
    '''
    client = MlflowClient()

    EXPERIMENT_ID = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

    query = "tags.mlflow.parentRunId = '{}'".format(parent_runid)
    stage_id = mlflow.search_runs(experiment_ids=[EXPERIMENT_ID], filter_string=query)
    stage_id = stage_id[stage_id['tags.mlflow.runName'] == stage]

    for runid in stage_id.run_id:
        if runid == id_to_retain:
            continue
        query = "tags.mlflow.parentRunId = '{}'".format(runid)
        child_ids = mlflow.search_runs(experiment_ids=[EXPERIMENT_ID], filter_string=query)
        for ch_runid in child_ids.run_id: #if stage has children e.g. Train - kfolds
            client.delete_run(ch_runid)
        client.delete_run(runid)