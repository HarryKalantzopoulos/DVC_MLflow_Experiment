# Version and Tracking experiments with dvc and mlflow
---

This is a demo to to version and track experiments, according to MLOps. As a Biomedical Engineer, I am performing a mockup experiment to show how can we use these two services to log information from all stages of a pipeline. I have already perform a data versioning on https://github.com/HarryKalantzopoulos/dvc_data_version, and by using dvc you can download some data to use. The source of the data is the PICAI-challenge.

For the demonstration I am using:
1. A local mlflow server, allong with a minio. I am using a docker image found in https://dagshub.com/rkchelebiev/mlops-dvc-mlflow, to deploy a local server.
2. A pipeline to segment the prostate from the images.
3. A segmentation Unet like architecture from https://github.com/yingkaisha/keras-unet-collection

Furthermore, except from collecting hyper-parameters which can be found inside *params.yaml*, usefull metadata and artifacts about data and code are collected, such as the md5 hashes of the input/output files, python packages version are collected (more in *code/return_md5.py*). To avoid 'failed' experiments to fill the space of mlflow, if a stage is reruning again for a experiment, the previous ones are deleted. To perform a new experiment, the user has to either change experiment bucket or chage *mlflow:name* inside  *params.yaml*.

A Demo of the process can be found in *notebooks/DEMO.ipynb*, even though the system OS shouldn't be a problem, I am performing the experiments on Windows 11 and WSL.

---
# Setting up local MLflow server
---
More information can be found in https://dagshub.com/rkchelebiev/mlops-dvc-mlflow
You can define the **.env** file to give your personal environment variables, in order to set up your services.

In this example we are using these urls:

Minio: http://127.0.0.1:9000/

MLflow: http://127.0.0.1:5000/

Then run:

```docker
docker compose up
```
---
# DVC and Mlflow
---
A demo can be found inside notebooks.
The organization of the working directory is based on Cookiecutter (https://drivendata.github.io/cookiecutter-data-science/).

**You will need to download the data from https://github.com/HarryKalantzopoulos/dvc_data_version**.

You only need .dvc and data.dvc to perform dvc pull:

```bash
git clone https://github.com/HarryKalantzopoulos/dvc_data_version.git .temp
mv .temp/.dvc ./.dvc
mv .temp/data.dvc ./data.dvc
rm -rf .temp
dvc pull
```

After preparing the pipeline (see code directory), dvc is used to track all the code and data changes. Each stage produces metadata which are stored locally in **DS_VERSION.json** and in **MLflow server**. The user can use *params.yaml* to name the experiments and choose other hyperparametes.

To set up your Mlflow server, use the values set in .env at **code.utils.mlflow_setup**.

If you use **dvc add stage** instead of **dvc run**, it will not run the stage and will create only dvc.yaml (registers stage to pipeline), eitherwise it will also create dvc.lock, which is a machine-readable format of dvc.yaml.

The stages are preprocess,prepare (kfold) and train.

e.g. preprocess stage:

```bash
dvc stage add -n md5_Preprocess \
    -p params.yaml:Preprocess.image_size,Preprocess.resample,Preprocess.maskcrop,Preprocess.8bit,mlflow.activate,mlflow.name \
    -d code/preprocess.py -d preprocess/images -d preprocess/masks \
    -O .temp/Preprocess.txt \
    python3  code/return_md5.py "Preprocess"
```
-n: name

-p: parameters of the stage found in params.yaml

-d: dependencies

-O: outputs but DVC will not store cache; if you want cache change to lowercase o

command line

dvc.yaml
```yaml
  Preprocess:
    cmd: python code/preprocess.py
    deps:
    - .temp/pipeline.txt
    - code/preprocess.py
    - data
    params:
    - Preprocess.8bit
    - Preprocess.image_size
    - Preprocess.maskcrop
    - Preprocess.resample
    - mlflow.activate
    - mlflow.name
    outs:
    - preprocess/dataset.csv:
        cache: false
    - preprocess/images:
        cache: false
    - preprocess/masks:
        cache: false
```

Inside dvc.lock, after a stage is run, the DVC will create the md5 hashes example:

dvc.lock
```yaml
  Preprocess:
    cmd: python code/preprocess.py
    deps:
    - path: .temp/pipeline.txt
      md5: 14cc740cd377a0fa586b0d318e7453d8
      size: 13
    - path: code/preprocess.py
      md5: 3e53b28ee8c2c83628bab585a2aa6d27
      size: 6385
    - path: data
      md5: ba30de71e034b2e63036d2d2f122e82a.dir
      size: 86051558
      nfiles: 20
    params:
      params.yaml:
        Preprocess.8bit: true
        Preprocess.image_size: 160
        Preprocess.maskcrop: true
        Preprocess.resample:
        - 3.0
        - 0.5
        - 0.5
        mlflow.activate: true
        mlflow.name: mydemopipeline
    outs:
    - path: preprocess/dataset.csv
      md5: 4cdfe936c5411d7df5d49fca57d7c625
      size: 947
    - path: preprocess/images
      md5: 8e4531ae96fda58c3f0cddf7bfd8b77c.dir
      size: 5262475
      nfiles: 10
    - path: preprocess/masks
      md5: 52281e8981938b8ab09752fff303bfea.dir
      size: 36601
      nfiles: 10
```
To keep track of these hashes, along with the versions of the used python packages, return_md5.py is used after each stage.

Since the order of the process depends on the dependencies and outputs, we define a hidden temp folder which stores plain txt files.

To run all the stages, use **dvc repro**. Afterwards, if a stage change (e.g. parameter), **dvc repro** will run only those stages that change. Here, the **dvc.yaml** and **dvc.lock** are given, so you run dvc repro after you install all the **requirements.txt**.

The MLflow tracking was organized as nested with parent the name of the experiment given in params.yaml. All the other stages are children and train stage has the kfold as children.

To avoid overflowing MLflow with failed experiments, as long as the name does not change, it is set to delete stage runs that already exist. Except if **dvc repro -f** (force) is used, then another experiment with the same name will be created.

Note that MLflow logs with the commit before the experiment is run, however the dvc related files (dvc.yaml, dvc.lock) can only be commited after the end of the experiment. 

---
# Model Registry
---
Finally, you can proceed to register your model either manually from MLflow or by using the code/register_model.py.

register_model.py will create and set the model for production.

---
# Dockerfile
---
Dockerfile exists if you want to run this example with docker, be sure to initialize git first.
```docker
docker build -t demo_dvc_mlflow .
docker run --network=host demo_dvc_mlflow
```
---
# Badges
---

![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)
![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04-orange.svg)](https://ubuntu.com/)
[![Windows](https://img.shields.io/badge/Windows-11-blue.svg)](https://www.microsoft.com/en-us/windows/)
![Docker](https://img.shields.io/badge/Docker-Containerization-blue)
![MinIO](https://img.shields.io/badge/MinIO-Storage%20Server-green)
![MLflow](https://img.shields.io/badge/MLflow-Tracking%20Server-blue)

