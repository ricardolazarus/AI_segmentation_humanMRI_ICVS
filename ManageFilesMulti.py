########################################################################
##  This function is part of the Rat and Human Brain Segmentation DL workflow
########################################################################
# Class to read, load and pre-process all nifti files
# Files will all be handled in full 3D or 2D
# Pre-processing includes only the intensity range normalization
########################################################################
# Mandatory inputs include directories for the original MRI brain images, ...
# ... for the images with classified tissues and the binary brain masks.
# Each subject must have the three different images.
########################################################################
# Created and edited by M. Rodrigues and R. Magalhães
########################################################################
import aux_functions
import os
import nibabel as nib
import numpy as np
from keras import utils


class myFiles:

    # Must by default initiate with directories, dimension (2D or 3D) and task (mask, segmentation, slices)
    def __init__(self, directory_mri, directory_classified, dimension, task, num_classes):
        self.nii = '.nii.gz'  # nifti file extension to help identify files
        # Here we will read only the files that have a nifti extension in the folder
        self.directory_mri = directory_mri
        self.directory_classified = directory_classified
        self.dimension = dimension
        self.task = task
        self.num_classes = num_classes

    def openlist(self):
        files_mri = []  # Variable to save all MRI images for both original MRI and binary masks
        files_classified = []
        files_bet = []  # and the brain extracted ground truth
        # get all files in the folder that have the correct extension
        files_list_MRI = sorted(
            [fn for fn in os.listdir(self.directory_mri) if self.nii in fn])
        files_list_classified = sorted(
            [fn for fn in os.listdir(self.directory_classified) if self.nii in fn])

        # all folders should have the same number of files
        if len(files_list_MRI) == len(files_list_classified):
            num_files = len(files_list_classified)
        else:  # if they don't have we will use the minimum size
            num_files = min(len(files_list_MRI), len(files_list_classified))

        # Will cycle through the entire array of MRI acquisitions to the get the matching files
        for i in range(num_files):
            aux_functions.update_progress("Loading Images:", i/num_files)
            # We will split the name according to the identifier "." to get the part that of the name
            # that allows us to match the different files of the same subject
            splits = files_list_MRI[i].split(".")
            name_mri = splits[0]
            image_mri = nib.load(self.directory_mri+"/" +
                                 files_list_MRI[i]).get_data()
            image_mri = aux_functions.normalize_image(image_mri)

            # Will search through all the classified files to find the one matching
            for name_classified in files_list_classified:  # the MRI one that was loaded
                if name_mri in name_classified:
                    files_mri.append(image_mri)
                    classified = (
                        nib.load(self.directory_classified+"/"+name_classified)).get_data()
                    files_classified.append(classified)
                    classified = np.asarray(classified)
                    # to create a brain extracted MRI (BET) we must assure we have binary images
                    mask = (classified > 0.5).astype(np.int_)
                    bet = np.multiply(image_mri, mask)
                    files_bet.append(bet)
                    break
        #convert all data to numpy arrays
        files_mri = np.asarray(files_mri)
        files_bet = np.asarray(files_bet)
        files_classified = np.asarray(files_classified)
        aux_functions.update_progress("Loading Images:", 1)

        if self.dimension == "3D":  # Prepare 3D data
            # The following dimensions need to be readjusted depending on the shape of the data being loaded
            # although this could be made automatic by reading the nifti file header or information
            # from the numpy array
            files_mri = np.reshape(
                files_mri, [files_mri.shape[0], files_mri.shape[1], files_mri.shape[2], files_mri.shape[3], 1])
            files_bet = np.reshape(
                files_bet, [files_bet.shape[0], files_bet.shape[1], files_bet.shape[2], files_bet.shape[3], 1])
            files_classified = np.reshape(
                files_classified, [files_classified.shape[0], files_classified.shape[1], files_classified.shape[2], files_classified.shape[3], 1])
            aux_functions.update_progress("Loading Images:", 1)
            # Next, will return different files depending on the task
            if self.task == "mask":
                return files_mri, files_classified
            elif self.task == "semantic":
                # the number here depends on how many tissues classes
                # we have, in this case we have 2 or 3 + background
                files_classified = utils.to_categorical(
                    files_classified, self.num_classes)
                return files_bet, files_classified
            else:
                print("ERROR, no proper task given")
                return None

        elif self.dimension == "2D":
            # Reshape it to be a set of 2D files
            # once again the shape is case dependent
            files_mri = np.concatenate(files_mri, 2)
            files_mri = np.rollaxis(files_mri, 2).reshape(
                files_mri.shape[2], files_mri.shape[0], files_mri.shape[1], 1)

            files_bet = np.concatenate(files_bet, 2)
            files_bet = np.rollaxis(files_bet, 2).reshape(
                files_bet.shape[2], files_bet.shape[0], files_bet.shape[1], 1)

            files_classified = np.concatenate(files_classified, 2)
            files_classified = np.rollaxis(files_classified, 2).reshape(
                files_classified.shape[2], files_classified.shape[0], files_classified.shape[1], 1)

            if self.task == "mask":
                return files_mri, files_classified
            elif self.task == "semantic":
                files_classified = utils.to_categorical(
                    files_classified, self.num_classes)
                return files_bet, files_classified

            elif self.task == "slices":  # if we want to do slice classification we will
                slicesMask = []          # build an array slice by slice to return
                count0 = 0
                count1 = 0
                for i in range(len(files_classified)):
                    if (np.max(files_classified[i]) == 0):
                        slicesMask.append(0)
                        count0 = count0+1
                    else:
                        slicesMask.append(1)
                        count1 = count1+1
                return files_mri, slicesMask
            else:
                print("ERROR, no proper task given")
                return None
        else:
            print("ERROR, no proper dimension given")
            return None
