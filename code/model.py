import tensorflow as tf
from keras_unet_collection import models
from keras_unet_collection.losses import dice,iou_seg
import yaml

def modelU():

    params = yaml.safe_load(open("params.yaml"))
    size = params["Preprocess"]["image_size"]

    params = params["model"]

    filters = params['filters']
    unet = params['architecture']
    loss= params['loss']
    opt= params['optimiser']
    metric= params['metric']
    Nseq = params['Number_inputs']
    Nlab = params['Number_labels']
    lact = params['layer_activation']
    act = params["activation"]
    dilation = params['dilation']

    #%%Params
    if lact=='LeakyReLU':
        lact = tf.keras.layers.LeakyReLU(alpha=0.1)

    if unet == 'Unet2D':
        model=models.unet_2d(input_size=(size,size,Nseq),filter_num=filters,n_labels=Nlab,
                   stack_num_down=2,stack_num_up=2,activation=lact,
                   output_activation=act,batch_norm=True,pool=True,
                   unpool=True,backbone=None,weights=None,freeze_backbone=True,
                   freeze_batch_norm=True,name=unet)

    elif unet == 'Attention_Unet':
        model=models.att_unet_2d(input_size=(size,size,Nseq),filter_num=filters,n_labels=Nlab,
                   stack_num_down=2,stack_num_up=2,activation=lact,attention='add',
                   output_activation=act,batch_norm=True,pool=False,
                   unpool=False,backbone=None,weights=None,freeze_backbone=True,
                   freeze_batch_norm=True,name=unet)

    elif unet == 'ResUnet':
        model=models.resunet_a_2d(input_size=(size,size,Nseq),dilation_num=dilation,filter_num=filters,n_labels=Nlab,
                  aspp_num_down=256,aspp_num_up=128,activation=lact,
                  output_activation=act,batch_norm=True,pool=True,
                  unpool=True,name=unet)

    #%% model

    if metric == "IoU":
        metric = tf.keras.metrics.OneHotIoU(num_classes=2, target_class_ids=[1],name=metric)

    if loss == 'dice':
        yt,yp = find_loss(y_true,y_pred)
        loss = dice(yt,yp)
    elif loss == 'iou_seg':
        yt,yp = find_loss(y_true,y_pred)
        loss = iou_seg(yt,yp)



    model.compile(loss=loss,
                optimizer=opt,
                metrics=[metric])

    return model


def find_loss(y_true, y_pred):
    y_pred=tf.where(y_pred[:,:,:,-1]>0.5,1,0)
    y_true=y_true[:,:,:,-1]
    return y_true,y_pred