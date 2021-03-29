import sys
import numpy as np
import os
import nibabel as nib

####################################    PROCESSING AUX FUNCTIONS     #########################################################


def update_progress(job_title, progress):
    """ Creates a progress bar
    Args:
        job_title (str): Name of the progress
        progress (float): Percentage of the progress bar
    """
    length = 30  # modify this to change the length
    block = int(round(length*progress))
    msg = "\r{0}: [{1}] {2}%".format(
        job_title, ">"*block + "-"*(length-block), round(progress*100, 2))
    sys.stdout.write(msg)
    sys.stdout.flush()
    if progress >= 1:
        msg += " DONE\r\n"
        sys.stdout.write(msg)
        sys.stdout.flush()


def split_train_evaluate(x_cases, y_cases, split_percentage):
    """Split to train and evaluation of the model the numpy cases
    Args:
        x_cases (np.ndarray): Cases in numpy format with the shape: (number_cases, x, y, z)
        y_cases (np.ndarray): Prediction segmentation masks
        split_percentage (float): Split value
    Returns:
        tuple: pair of x train and evaluate, pair of y train and evaluate
    """

    n_split = int(x_cases.shape[0] * split_percentage)
    x_training = x_cases[:n_split]
    y_training = y_cases[:n_split]
    x_evaluate = x_cases[n_split:]
    y_evaluate = y_cases[n_split:]

    return x_training, y_training, x_evaluate, y_evaluate


def correct_images_size(image, size):
    """Update images size by adding empty columns.
    Args:
        image (np.ndarray): Numpy to update shape
        size (int): Image size on all the axis. If is 256 then the 3D image will have (256,256,256).
    Returns:
        np.ndarray: New shape of images
    """

    if (size != image.shape[0]):
        image = np.append(image, np.zeros((size - image.shape[0], image.shape[1], image.shape[2])),axis=0)
    if (size != image.shape[1]):
        image = np.append(image,np.zeros((image.shape[0], size - image.shape[1], image.shape[2])),axis=1)
    if (size != image.shape[2]):
        image = np.append(image,np.zeros((image.shape[0], image.shape[1], size - image.shape[2])),axis=2)

    return image


def normalize_image(image):
    """Pre-process image by normalization between 0 and 1.
    Args:
        image (np.ndarray): Numpy image.
    Returns:
        np.ndarray: Image processed from MRI file
    """
    image = image.get_data()
    # NORMALIZATION
    image = np.floor(image)
    image /= np.max(image)  # division by the maximum

    return image


def random_shuffle(x_training, x_evaluate, y_training, y_evaluate):
    """Random shuffle in x and y before training.
    Args:
        x_training ([type]): Training cases images in numpy
        y_training ([type]): Evaluate cases images in numpy
        x_prediction ([type]): Training cases predictions in numpy
        y_prediction ([type]): Evaluate cases predictions in numpy
    Returns:
        tuple: x_training, x_evaluate, y_training, y_evaluate with random order.
    """
    idx = np.random.permutation(len(x_training)) 
    x_training, x_evaluate, y_training, y_evaluate = x_training[idx], x_evaluate[idx], y_training[idx], y_evaluate[idx]
    return x_training, x_evaluate, y_training, y_evaluate


def check_folders(save_dir):
    """  Make sure we have folder structure we are using
    Args:
        save_dir (str): saving directory to be checked
    """
    if not os.path.exists(os.path.join(save_dir, 'logs')):
        os.makedirs(os.path.join(save_dir, 'logs'))
    if not os.path.exists(os.path.join(save_dir, 'logs/model')):
        os.makedirs(os.path.join(save_dir, 'logs/model'))
    if not os.path.exists(os.path.join(save_dir, 'bestvalueslasttrain')):
        os.makedirs(os.path.join(save_dir, 'bestvalueslasttrain'))


####################################    EVALUATION AUX FUNCTIONS     #########################################################


def evaluate_binary_mask(ground_mask, generated_mask):
    """Evaluates binary generated mask with ground truth mask.
    Args:
        ground_mask (np.ndarray): Ground truth mask
        generated_mask (np.ndarray): Generated mask to compare with ground truth
    Returns:
        tuple: Values of accuracy, sensitivity, specificity and dice
    """

    ground_mask = np.squeeze(ground_mask, ground_mask.ndim - 1)
    generated_mask = np.squeeze(generated_mask, generated_mask.ndim - 1)
    y_true_f = ground_mask.flatten()
    y_pred_f = generated_mask.flatten()
    tp = np.round(np.sum(y_pred_f*y_true_f), 0) # true positives
    tn = np.round(np.sum((1 - y_pred_f)*(1 - y_true_f)), 0) # true negatives
    fp = np.round(np.sum(y_pred_f*(1 - y_true_f)), 0) # false positives
    fn = np.round(np.sum((1 - y_pred_f)*y_true_f), 0) # false negatives
    accuracy = (tp + tn)/(tp + tn + fp + fn)
    sensitivity = tp/(tp + fn)
    specificity = tn/(tn + fp)
    dsc = (2 * tp)/(2*tp + fp + fn)  # Dice coefficient

    return accuracy, sensitivity, specificity, dsc


def evaluate_multiclass_mask(ground_mask, generated_mask, number_classes):
    """Evaluates multiclass generated mask with ground truth mask.
    Args:
        ground_mask (np.ndarray): Ground truth mask
        generated_mask (np.ndarray): Generated mask to compare with ground truth
        number_classes (int): Number of classes to compare
    Returns:
        tuple: Class number, Values of accuracy, sensitivity, specificity and dice
    """

    total_tp = total_tn = total_fp = total_fn = 0
    values = []
    ground_mask = np.moveaxis(ground_mask, ground_mask.ndim-1, 0)
    generated_mask = np.moveaxis(generated_mask, ground_mask.ndim-1, 0)
    for i in range(1, number_classes+1):
        y_true_f = ground_mask[i].flatten()
        y_pred_f = generated_mask[i].flatten()
        tp = np.round(np.sum(y_pred_f*y_true_f), 0)
        total_tp = total_tp + tp
        tn = np.round(np.sum((1-y_pred_f) * (1-y_true_f)), 0)
        total_tn = total_tn + tn
        fp = np.round(np.sum(y_pred_f * (1-y_true_f)), 0)
        total_fp = total_fp + fp
        fn = np.round(np.sum((1-y_pred_f)*y_true_f), 0)
        total_fn = total_fn + fn
        accuracy = (tp + tn)/(tp + tn + fp + fn) #calculate for each tissue map
        sensitivity = tp/(tp + fn)
        specificity = tn/(tn + fp)
        dsc = (2*tp)/(2*tp + fp + fn)
        values.append([i, accuracy, sensitivity, specificity, dsc])

    accuracy = (total_tp + total_tn) / \
        (total_tp + total_tn + total_fp + total_fn) #calculate for the total
    sensitivity = total_tp/(total_tp + total_fn)
    specificity = total_tn/(total_tn + total_fp)
    dsc = (2*total_tp)/(2*total_tp + total_fp + total_fn)
    values.append([i + 1, accuracy, sensitivity, specificity, dsc])
    return values