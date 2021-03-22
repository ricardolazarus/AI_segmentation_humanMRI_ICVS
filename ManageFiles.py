import os

import NumpyImage
import helpingtools
import nibabel as nib
import numpy as np
from keras import utils

class myFiles2D:

    def __init__(self, directoryBrains):
        self.directoryBrains = directoryBrains
        self.numberOfFiles = []
        self.shapex = 256
        self.shapey = 256
        self.shapez = 256
        self.CLASS_NUMBER = 3
    def openlist(self):
        mri = []
        mask = []
        wgm = []
        wgm4 = []
        mribet = []
        mri_notzero = []
        mask_notzero = []
        wgm_notzero = []
        mribet_notzero = []
        
        files_list = sorted(os.listdir(self.directoryBrains))
        self.numberOfFiles = int(len(files_list) / 3)
        for i in range(0 , self.numberOfFiles * 3 , 3):
            helpingtools.update_progress("Loading and Pre-Processing MRI and Ground truth data", i / (self.numberOfFiles * 3))
            filepatht1 = self.directoryBrains + "/" + files_list[i]
            filepathwmg = self.directoryBrains + "/" + files_list[i + 2]
            filepathmask = self.directoryBrains + "/" + files_list[i + 1]
            
            numpyt1 = nib.load(filepatht1)
            numpyt1 = NumpyImage.myNumpy(numpyt1)
            numpyt1 = numpyt1.preProcessing()
            numpyWMG = nib.load(filepathwmg).get_data()
            numpymask = (nib.load(filepathmask)).get_data()
            numpyt1bet = np.multiply(numpyt1,numpymask)
            mri.append(numpyt1)
            mask.append(numpymask)
            wgm.append(numpyWMG)
            mribet.append(numpyt1bet)
            
        mri = np.concatenate(mri, 2)
        mri = np.rollaxis(mri, 2).reshape(mri.shape[2], mri.shape[0], mri.shape[1], 1)
        
        mask = np.concatenate(mask, 2)
        mask = np.rollaxis(mask, 2).reshape(mask.shape[2], mask.shape[0], mask.shape[1], 1)
            
        wgm = np.concatenate(wgm, 2)
        wgm = np.rollaxis(wgm, 2).reshape(wgm.shape[2], wgm.shape[0], wgm.shape[1], 1)
        
        mribet = np.concatenate(mribet, 2)
        mribet = np.rollaxis(mribet, 2).reshape(mribet.shape[2], mribet.shape[0], mribet.shape[1], 1)
        print(mribet.shape)
        for i in range(0, len(wgm)):
            if np.count_nonzero(wgm[i]) != 0:
                mri_notzero.append(mri[i])
                mask_notzero.append(mask[i])
                wgm_notzero.append(wgm[i])
                mribet_notzero.append(mribet[i])
                
        print(np.asarray(mribet_notzero).shape)
        mri_notzero=np.asarray(mri_notzero)
        mask_notzero=np.asarray(mask_notzero)
        wgm_notzero=np.asarray(wgm_notzero)
        mribet_notzero=np.asarray(mribet_notzero)
        wgm_notzero = utils.to_categorical(wgm_notzero, 3)
        return mri_notzero, mask_notzero, wgm_notzero, mribet_notzero

