# Version and Tracking experiments with dvc and mlflow
---
The main targer
This is a demo to to version and track experiments,according to MLOps. As a Biomedical Engineer, I am performing a mockup experiment to show how can we use these two services to log informations from all stages of a pipeline. I have already perform a data versioning on https://github.com/HarryKalantzopoulos/dvc_data_version, and by using dvc you can download some data to use. The source of the data is the PICAI-challenge.

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

and all are set up.

---

# Badges
---
| Python version | Badge |
| 3.9            | ![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg) |
| 3.10           | ![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg) |