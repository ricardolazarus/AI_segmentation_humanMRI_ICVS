import importlib
import os
import time
from keras import losses, optimizers
from models import unet_2d
import ManageFiles
import psutil
import writegraphs
import sys
from train import dice_coef_loss
import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping#LR
from keras.models import load_model
import nibabel as nib
import time
from loss_functions import Precision, FPR, FNR, specificity, sensitivity, Tversky_loss, dice_coef, dice_coef_multilabel_tissue_matters, Tversky_multilabel, sensitivity_background, sensitivity_tissue2, sensitivity_tissue1
from aux_functions import split_train_evaluate, evaluate_binary_mask, evaluate_multiclass_mask

IMAGE_PATH = ''
SPLIT_PERCENTAGE = 0.8 # Define percentage of Training images, the rest will be evaluation#

def run_all_images_wgm(predictions,masks_evaluate):
    total_values=[[0,0,0,0,0],[1,0,0,0,0],[2,0,0,0,0]]
    for n in range(0,len(predictions)):
        
        gen_mask=np.asarray(predictions[n]) #get the ground truth
        gen_mask=np.squeeze(gen_mask,0)
        gt_mask=np.asarray(masks_evaluate[n]) #get the generated masks
        values=evaluate_multiclass_mask(gt_mask, gen_mask,3)
        for colormap in range(len(values)):
            total_values[colormap][1]=total_values[colormap][1]+values[colormap][1]
            total_values[colormap][2]=total_values[colormap][2]+values[colormap][2]
            total_values[colormap][3]=total_values[colormap][3]+values[colormap][3]
            total_values[colormap][4]=total_values[colormap][4]+values[colormap][4]
    n=n+1
    
    for colormap in range(len(total_values)):
        print("#########################################")
        print("########### Final Results ###############")
        print("#        Accuracy: " + str(np.round(total_values[colormap][1] / n,5)) + "    #")
        print("#     Sensibility: " + str(np.round(total_values[colormap][2] / n,5)) + "     #")
        print("#     Specificity: " + str(np.round(total_values[colormap][3] / n,5)) + "    #")
        print("#             DSC: " + str(np.round(total_values[colormap][4] / n,5)) + "     #")
        print("#########################################")

def run_all_images_bin(predictions,masks_evaluate):
    sum_acc=sum_sen=sum_spe=sum_dsc=0
    for n in range(0,len(predictions)):
        gen_mask=np.asarray(predictions[n]) #get the ground truth
        gt_mask=np.asarray(masks_evaluate[n]) #get the generated masks
        acc,sens,spe,dsc=evaluate_binary_mask(gt_mask, gen_mask)
       	sum_acc=sum_acc+acc
       	sum_sen=sum_sen+sens
       	sum_spe=sum_spe+spe
       	sum_dsc=sum_dsc+dsc
    n=n+1
    print("########### Final Results ###############")
    print("#        Accuracy: " + str(np.round(sum_acc / n,5)) + "    #")
    print("#     Sensibility: " + str(np.round(sum_sen / n,5)) + "     #")
    print("#     Specificity: " + str(np.round(sum_spe / n,5)) + "    #")
    print("#             DSC: " + str(np.round(sum_dsc / n,5)) + "     #")
    print("#########################################")


def predict_bin (mris_evaluate,model,predictions,ref_shape):
    for file in mris_evaluate:
        file2 = np.reshape(file,ref_shape.shape)
        prediction = model.predict(file2,batch_size=1)
        prediction = np.reshape(prediction,file.shape)
        prediction = np.round(prediction)
        predictions.append(prediction)
    return predictions

def predict_multi (mris_evaluate,model,predictions,ref_shape_in,ref_shape_out):
    for file in mris_evaluate:
        file = file.reshape(ref_shape_in.shape)
        prediction = model.predict(file,batch_size=1)
        prediction = np.reshape(prediction,ref_shape_out.shape)
        prediction = np.round(prediction)
        predictions.append(prediction)
    return predictions

def save_slice(slic,i):
    dir_slices=""
    file=nib.load(dir_slices + "file.nii.gz")
    affine=file.affine
    save=nib.Nifti1Image(slic,affine)
    nib.save(save,dir_slices+'/'+str(i)+'gt.nii.gz')

if __name__ == '__main__':
    dire=''
    mod=sys.argv[1]
    modelname=sys.argv[2]
    predictions=[]
    ###Load the model we want to evaluate
    print("Running model: ",modelname)
    
    ###Load the images we want to test. Right now it's loading all images possible and then keeping only we want. This can be changed in the future for efficiency
    files = ManageFiles.myFiles2D(IMAGE_PATH)
    mri, mask, wgm, mribet = files.openlist()
    if mod=="mask":
        modelfile=dire+'/'+mod+'/'+modelname+'.h5'
        model=load_model(modelfile, custom_objects={'dice_coef_loss': dice_coef_loss,'Precision': Precision,'FPR': FPR, 'FNR': FNR, 'specificity': specificity, 'sensitivity': sensitivity})
        mris_training, masks_training, mris_evaluate,masks_evaluate=definetrainevaluate(mri,mask)
        ref_shape=mris_evaluate[1,:,:,:]
        ref_shape=ref_shape.reshape(1,256,256,1)
        predictions=predict_bin(mris_evaluate,model,predictions,ref_shape)
        #save_nifti(predictions,masks_evaluate,dim,modelname,dire)
        run_all_images_bin(predictions,masks_evaluate) 
    if mod=="semantic":
        modelfile=dire+'/'+mod+'/'+modelname+'.h5'
        model=load_model(modelfile, custom_objects={'dice_coef_loss': dice_coef_loss, 'Precision': Precision,'FPR': FPR, 'FNR': FNR, 'specificity': specificity, 'sensitivity': sensitivity,'dice_coef_multilabel_tissue_matters':dice_coef_multilabel_tissue_matters, 'sensitivity_background':sensitivity_background, 'sensitivity_tissue1':sensitivity_tissue1, 'sensitivity_tissue2':sensitivity_tissue2,'Tversky_multilabel': Tversky_multilabel})
        mris_training, masks_training, mris_evaluate,masks_evaluate=definetrainevaluate(mribet,wgm)
        ref_shape_in=mris_evaluate[1,:,:,:]
        ref_shape_in=ref_shape_in.reshape(1,256,256,1)
        ref_shape_out=masks_evaluate[1,:,:,:]
        ref_shape_out=ref_shape_out.reshape(1,256,256,3)
        predictions=predict_multi(mris_evaluate,model,predictions,ref_shape_in,ref_shape_out)
        run_all_images_wgm(predictions,masks_evaluate)
    