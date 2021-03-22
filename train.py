import importlib
import os
import time
from keras import losses, optimizers
import Arq_SegBrain, Arq_WGM
import ManageFiles
import loss_functions
import psutil
import writegraphs
import sys
import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping#LR
import nibabel as nib


##################  BEGIN VARIABLES  #################################
# Prepare files info
CLASS_NUMBER=3
N_FILES = 54  # Max number of files = 54
IMAGE_PATH = ''
SPLIT_PERCENTAGE = 0.8 # Define percentage of Training images, the rest will be evaluation#
# Training info
EPOCHS = 1000
BATCH_SIZE = 5
# Define parameters of optmizer
lrate = 0.0002
decay = lrate / 200
momentum = 0.9
VERBOSE_TRAIN = 1  # 0-only text, 1-generating images
# See images (not working currently)
VERBOSE_imageGenerator = 1  # 0-no train comments, 1-text each epoch
PRINT_NUMBER = 20  # Choose a number to print in a multiple epoch
SELECTED_LAYERS = [3, 6, 10, 20, 30, 35, 37]  # Select only some Layers to print featuremaps
#################  ENDING VARIABLES  #################################

def definetrainevaluate(mris, masks):  # split train data and evaluate data
    split_len = int(len(mris) * SPLIT_PERCENTAGE)
    mris_training = mris[:split_len]
    masks_training = masks[:split_len]
    mris_evaluate = mris[split_len:]
    masks_evaluate = masks[split_len:]
    return mris_training, masks_training, mris_evaluate, masks_evaluate

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def compiletrain(mris_training, masks_training, model):
    print("\n ########### TRAINING ############\n")##
    # define metrics
    VS=int(len(mris_training)/BATCH_SIZE)
    SE=int((len(mris_training)*0.2)/BATCH_SIZE)
    if TYPEOFDEEPLEARNING=="mask":
        MODEL_SAVE=""
        all_metrics = ['accuracy', loss_functions.Precision, loss_functions.FPR, loss_functions.FNR, loss_functions.specificity]
        sgd = optimizers.SGD(lr=lrate, momentum=0.9, decay=decay) #, nesterov=False)
        model.compile(loss=loss_functions.dice_coef_loss, optimizer='sgd', metrics=all_metrics)
    elif TYPEOFDEEPLEARNING=="semantic":
        MODEL_SAVE=""
        all_metrics = ['accuracy', loss_functions.sensitivity_background, loss_functions.sensitivity_tissue1, loss_functions.sensitivity_tissue2, loss_functions.dice_coef_multilabel_tissue_matters,loss_functions.Tversky_multilabel]
        sgd = optimizers.SGD(lr=lrate, momentum=0.9, decay=decay)#, nesterov=False)
        model.compile(loss=loss_functions.dice_coef_multilabel_tissue_matters, optimizer=sgd, metrics=all_metrics)
        #model.compile(loss=loss_functions.Tversky_multilabel, optimizer=sgd, metrics=all_metrics)

    # Save tensorboard
    #tensorboard = writegraphs.TrainValTensorBoard(write_graph=False)
    # Save best models
    m1 = ModelCheckpoint(MODEL_SAVE + "/val_loss.h5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    m2 = ModelCheckpoint(MODEL_SAVE + "/val_acc.h5", monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    m3 = ModelCheckpoint(MODEL_SAVE + "/bestweights_vall_acc.hdf5", monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    m4 = ModelCheckpoint(MODEL_SAVE + "/bestweights_vall_loss.hdf5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    # Early Stoppping#
    e1 = EarlyStopping(monitor='val_loss', min_delta=0, patience=80, verbose=0, mode='auto')
    e2 = EarlyStopping(monitor='val_acc', min_delta=0, patience=80, verbose=0, mode='auto')
    # Fit model with images, begin training
    history = model.fit(mris_training, masks_training, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_split=0.2, initial_epoch=0, callbacks=[m1, m2, m3, m4, e1, e2], shuffle=True)
    # save final Model
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model.save(MODEL_SAVE + timestamp + ".h5")
    return model, history


def evaluate(mris_evaluate, masks_evaluate, model, history):
    p = psutil.Process()
    cpu_time = p.cpu_times()[0]
    mem = p.memory_info()[0]
    print("###########Evaluation############\n")
    scores = model.evaluate(mris_evaluate, masks_evaluate)
    print("\n Metric: %s: %.2f%%\n" % (model.metrics_names[1], scores[1] * 100))
    with open("", "a") as myfile:
        myfile.write(",")
        myfile.write("\"" + ascii(model).strip() + "\",")
        myfile.write("{},".format(history.history['loss'][-1]))
        myfile.write("{},".format(history.history['acc'][-1]))
        myfile.write("{},".format(mem))
        myfile.write("{},\n".format(cpu_time))


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


def limit_mem():
    K.get_session().close()
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))

def save_slice(slic,i):
    dir_slices=""
    file=nib.load(dir_slices + "file.nii.gz")
    affine=file.affine
    save=nib.Nifti1Image(slic,affine)
    nib.save(save,dir_slices+'/'+str(i)+'gt.nii.gz')
    
    
if __name__ == '__main__':
    TYPEOFDEEPLEARNING = sys.argv[1]
    set_keras_backend("tensorflow")
    files_list = sorted(os.listdir(IMAGE_PATH))
    print("There are " + str(int(len(files_list) / 3)) + " subjects. \n")
    
    if TYPEOFDEEPLEARNING == "mask":  # Segment only brain
        files = ManageFiles.myFiles2D(IMAGE_PATH)
        mris, masks, wgms, mrisbet = files.openlist()
        mris_training, masks_training, mris_evaluate, masks_evaluate = definetrainevaluate(mris, masks)
        layers=Arq_SegBrain.my2DUnet()
        model=layers.create_DL
        model, history = compiletrain(mris_training, masks_training, model)
    elif TYPEOFDEEPLEARNING == "semantic":  # Segment White and Gray Matter
        files = ManageFiles.myFiles2D(IMAGE_PATH)
        mri, mask, wgm, mribet = files.openlist()
        mris_training, masks_training, mris_evaluate, masks_evaluate = definetrainevaluate(mribet, wgm)
        layer = Arq_WGM.my2DUnet(CLASS_NUMBER)
        model = layer.create_DL
        model, history = compiletrain(mris_training, masks_training, model)
    
    elif TYPEOFDEEPLEARNING == "test":  # Segment only brain
        files = ManageFiles.myFiles2D(IMAGE_PATH)
        mris, masks, wgms, mrisbet = files.openlist()
        
        slic=np.dot(np.asarray(mrisbet[1,:,:,:]),100)
        slic=np.squeeze(slic)
        save_slice(slic,1)
        slic=np.dot(np.asarray(wgms[1,:,:,:]),100)
        slic=np.squeeze(slic)
        save_slice(slic,2)
        slic=np.dot(np.asarray(mrisbet[30,:,:,:]),100)
        slic=np.squeeze(slic)
        save_slice(slic,3)
        slic=np.dot(np.asarray(wgms[30,:,:,:]),100)
        slic=np.squeeze(slic)
        save_slice(slic,4)
        slic=np.dot(np.asarray(mrisbet[100,:,:,:]),100)
        slic=np.squeeze(slic)
        save_slice(slic,5)
        slic=np.dot(np.asarray(wgms[100,:,:,:]),100)
        slic=np.squeeze(slic)
        save_slice(slic,6)
        slic=np.dot(np.asarray(mrisbet[301,:,:,:]),100)
        slic=np.squeeze(slic)
        save_slice(slic,7)
        slic=np.dot(np.asarray(wgms[301,:,:,:]),100)
        slic=np.squeeze(slic)
        save_slice(slic,8)
        
    ################################         TRAINING DEEP LEARNING         ##############################
    #model, history = compiletrain(mris_training, masks_training, model)
    #evaluate(mris_evaluate, masks_evaluate, model, history)
    #print_history_accuracy(history)
    #print_history_loss(history)
