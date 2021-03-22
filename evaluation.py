import importlib
import os
import time
from keras import losses, optimizers
import Arq_SegBrain , Arq_WGM
import ManageFiles
import loss_functions
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

IMAGE_PATH = ''
SPLIT_PERCENTAGE = 0.8 # Define percentage of Training images, the rest will be evaluation#

def definetrainevaluate(mris, masks):  # split train data and evaluate data
    split_len = int(len(mris) * SPLIT_PERCENTAGE)
    mris_training = mris[:split_len]
    masks_training = masks[:split_len]
    mris_evaluate = mris[split_len:]
    masks_evaluate = masks[split_len:]
    return mris_training, masks_training, mris_evaluate, masks_evaluate


def evaluate_bin_one_image(ground_mask, generated_mask,n):
    ground_mask=np.squeeze(ground_mask,ground_mask.ndim-1)
    generated_mask=np.squeeze(generated_mask,generated_mask.ndim-1)
    if n==2065:
        save_slice(generated_mask,4)
    y_true_f = ground_mask.flatten()
    y_pred_f = generated_mask.flatten()
    TP = np.round(np.sum(y_pred_f*y_true_f),0)
    TN = np.round(np.sum((1-y_pred_f)*(1-y_true_f)),0)
    FP = np.round(np.sum(y_pred_f*(1-y_true_f)),0)
    FN = np.round(np.sum((1-y_pred_f)*y_true_f),0)
    accuracy=(TP+TN)/(TP+TN+FP+FN)
    sensitivity=TP/(TP+FN)
    specificity=TN/(TN+FP)
    dsc=(2*TP)/(2*TP+FP+FN) #Dice coefficient
    return accuracy, sensitivity, specificity,dsc

def evaluate_multiclasse_one_image(ground_mask,generated_mask,number_classes):
    TP=TN=FP=FN=0
    TPt=TNt=FPt=FNt=0
    values=[]
    ground_mask = np.moveaxis(ground_mask,ground_mask.ndim-1,0)
    generated_mask = np.moveaxis(generated_mask,ground_mask.ndim-1,0)
    for i in range (1,number_classes):
        y_true_f = ground_mask[i].flatten()
        y_pred_f = generated_mask[i].flatten()
        TP = np.round(np.sum(y_pred_f*y_true_f),0)
        TPt=TPt+TP
        TN = np.round(np.sum((1-y_pred_f)*(1-y_true_f)),0)
        TNt=TNt+TN
        FP = np.round(np.sum(y_pred_f*(1-y_true_f)),0)
        FPt=FPt+FP
        FN = np.round(np.sum((1-y_pred_f)*y_true_f),0)
        FNt=FNt+FN
        if TP == 0:
            TP = 1
        accuracy=(TP+TN)/(TP+TN+FP+FN)
        sensitivity=TP/(TP+FN)
        specificity=TN/(TN+FP)
        dsc=(2*TP)/(2*TP+FP+FN)
        values.append([i,accuracy,sensitivity,specificity,dsc])
        
    accuracy=(TPt+TNt)/(TPt+TNt+FPt+FNt)
    sensitivity=TPt/(TPt+FNt)
    specificity=TNt/(TNt+FPt)
    dsc=(2*TPt)/(2*TPt+FPt+FNt)
    values.append([i+1,accuracy,sensitivity,specificity,dsc])
    return values

def run_all_images_wgm(predictions,masks_evaluate):
    total_values=[[0,0,0,0,0],[1,0,0,0,0],[2,0,0,0,0]]
    for n in range(0,len(predictions)):
        
        gen_mask=np.asarray(predictions[n]) #get the ground truth
        gen_mask=np.squeeze(gen_mask,0)
        gt_mask=np.asarray(masks_evaluate[n]) #get the generated masks
        values=evaluate_multiclasse_one_image(gt_mask, gen_mask,3)
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
        acc,sens,spe,dsc=evaluate_bin_one_image(gt_mask, gen_mask,n)
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

    
def correct_images_size(numpyt1,size):
    
    if (size != numpyt1.shape[1]):
        numpyt1 = np.append(numpyt1,
                            np.zeros((numpyt1.shape[0], size - numpyt1.shape[1], numpyt1.shape[2])),
                            axis=1)
    if (size != numpyt1.shape[2]):
        numpyt1 = np.append(numpyt1,
                            np.zeros((numpyt1.shape[0], numpyt1.shape[1], size - numpyt1.shape[2])),
                            axis=2)
    if (size != numpyt1.shape[0]):
        numpyt1 = np.append(numpyt1,
                            np.zeros((size - numpyt1.shape[0], numpyt1.shape[1], numpyt1.shape[2])),
                            axis=0)
    return numpyt1


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
        
    if mod=="save1":
        predictions1=[]
        modelfile=dire+'/semantic/'+modelname+'.h5'
        model1=load_model(modelfile, custom_objects={'dice_coef_loss': dice_coef_loss, 'Precision': Precision,'FPR': FPR, 'FNR': FNR, 'specificity': specificity, 'sensitivity': sensitivity,'dice_coef_multilabel_tissue_matters':dice_coef_multilabel_tissue_matters, 'sensitivity_background':sensitivity_background, 'sensitivity_tissue1':sensitivity_tissue1, 'sensitivity_tissue2':sensitivity_tissue2,'Tversky_multilabel': Tversky_multilabel})
        mris_training, masks_training, mris_evaluate,masks_evaluate=definetrainevaluate(mribet,wgm)
        ref_shape_in=mris_evaluate[1,:,:,:]
        ref_shape_in=ref_shape_in.reshape(1,256,256,1)
        ref_shape_out=masks_evaluate[1,:,:,:]
        ref_shape_out=ref_shape_out.reshape(1,256,256,3)
        predictions=predict_multi(mris_evaluate,model1,predictions,ref_shape_in,ref_shape_out)
        predictions=np.squeeze(np.asarray(predictions),1)
        print(np.asarray(predictions).shape)
        print(np.asarray(mris_evaluate).shape)
        print(np.asarray(masks_evaluate).shape)
        slic=np.dot(np.asarray(predictions[2065,:,:,:]),100)
        slic=np.argmax(slic,2)
        slic=np.squeeze(slic)
        save_slice(slic,1)
        slic=np.dot(np.asarray(masks_evaluate[2065,:,:,:]),100)
        slic=np.argmax(slic,2)
        slic=np.squeeze(slic)
        save_slice(slic,3)
        
        slic=np.dot(np.asarray(mris_evaluate[2065,:,:,:]),100)
        slic=np.squeeze(slic)
        save_slice(slic,2)
        
    if mod=="save2":
        predictions=[]
        modelfile=dire+'/mask/'+modelname+'.h5'
        model2=load_model(modelfile, custom_objects={'dice_coef_loss': dice_coef_loss,'Precision': Precision,'FPR': FPR, 'FNR': FNR, 'specificity': specificity, 'sensitivity': sensitivity})
        mris_training, masks_tr, mris_evaluate,masks_ev=definetrainevaluate(mribet,mask)
        ref_shape=mris_evaluate[1,:,:,:]
        ref_shape=ref_shape.reshape(1,256,256,1)
        predictions=predict_bin(mris_evaluate,model2,predictions,ref_shape)
        
        #save_slice(generated_mask,4)
        #save_slice(ground_mask,5)