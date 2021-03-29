""" Train DL models for 2D convolutions """
import importlib
import os
import time
import sys

from keras import losses, optimizers
import models
import ManageFilesMulti
import loss_functions
import psutil
import writegraphs

import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
import nibabel as nib
from aux_functions import split_train_evaluate


def define_optimizer(optimizer, learning_rate=0.0003, decay_rate=0.0003/200, moment = 0.9):
    """ Define the optimizer and loss to be used
    Args:
        optimizer (str): Optimizer name
        learning_rate (float, optional): Value of learning rate. Defaults to 0.0003.
        decay_rate (float, optional): Value of decay. Defaults to 0.0000015.
    Returns:
        [type]: [description]
    """
    if optimizer == "adam":
        function_optmizer = optimizers.Adam(lr=learning_rate, decay=decay_rate)
    elif optimizer == "sgd":
        function_optmizer = optimizers.SGD(
            lr=learning_rate, decay=decay_rate, momentum=moment)
    print("Optimizer: "+str(function_optmizer))
    return function_optmizer

def compiletrain(mris_training, masks_training, model, function_optmizer, shuffle, loss, all_metrics, save_dir, epoch = 1000, batch_size=5):
    """ Train model and save checkpoints
    Args:
        mris_training (np.ndarray): MRI cases images
        masks_training (np.ndarray): Masks ground truth of mris_training cases
        model: Deep learning model to be trained
        function_optmizer: Optimizer function chosen
        shuffle (bool): Add shuffle cases
        loss: Loss function chosen
        all_metrics (list): List of all metrics chosen to be followed during training
        save_dir (str): Path to training info.
    Returns:
    model, history: Trained model and history
    """
    
    print("\n ########### TRAINING ############\n")
    model.compile(loss=loss, optimizer=function_optmizer, metrics=all_metrics)

    # Save best models
    m1 = ModelCheckpoint(save_dir + "/logs/" + "val_loss.h5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    m2 = ModelCheckpoint(save_dir + "/logs/" + "val_acc.h5", monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    m3 = ModelCheckpoint(save_dir + "/logs/" + "bestweights_vall_acc.hdf5", monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    m4 = ModelCheckpoint(save_dir + "/logs/" + "bestweights_vall_loss.hdf5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    # Early Stoppping#
    e1 = EarlyStopping(monitor='val_loss', min_delta=0, patience=80, verbose=0, mode='auto')
    e2 = EarlyStopping(monitor='val_acc', min_delta=0, patience=80, verbose=0, mode='auto')
    # Fit model with images, begin training
    history = model.fit(mris_training, masks_training, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.2, initial_epoch=0, callbacks=[m1, m2, m3, m4, e1, e2], shuffle=shuffle)
    # save final Model
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model.save(save_dir +"/logs/model/" + timestamp + ".h5")
    return model, history

def set_keras_backend(backend):
    print("A acertar o backend e libertar memoria da grafica")
    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        importlib.reload(K)
        assert K.backend() == backend
    if backend == "tensorflow":
        K.get_session().close()
        cfg = K.tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        # cfg.gpu_options.allocator_type = 'BFC'
        K.set_session(K.tf.Session(config=cfg))
        
if __name__ == '__main__':
    task = sys.argv[1]
    save_dir = sys.argv[2]
    set_keras_backend("tensorflow")
    #setup the path to the different images
    image_path = ''
    mask_path = ''
    semantic_path = ''

    shuffle=True
    factor=0.744
    function_optmizer = define_optimizer("adam")
    model = models.unet_2d(num_classes=1) if task=="mask" else models.unet_2d(num_classes=3)
    files = ManageFilesMulti.myFiles(image_path,mask_path,'2D',task,1)
    mris, mask = files.openlist()
    mris_training, masks_training, mris_evaluate, masks_evaluate = split_train_evaluate(mris, masks, factor)
    if task == "mask":  # Segment only brain
        all_metrics = ['accuracy', loss_functions.Precision, loss_functions.FPR, loss_functions.FNR, loss_functions.specificity]
        loss=loss_functions.dice_coef_loss

    elif task == "semantic":  # Segment White and Gray Matter
        all_metrics = ['accuracy', loss_functions.sensitivity_background, loss_functions.sensitivity_tissue1, 
            loss_functions.sensitivity_tissue2, loss_functions.dice_coef_multilabel_tissue_matters,loss_functions.Tversky_multilabel]
        loss=loss_functions.dice_coef_multilabel2D
        
    ################################         TRAINING DEEP LEARNING         ##############################
    model, history = compiletrain(mris_training,
         masks_training, model, function_optmizer,
         shuffle, loss,all_metrics,save_dir)

