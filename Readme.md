Deep Learning to segment different brain tissues
In this project, the main objective is to segment different brain tissues in Rats (With 2D and 3D DL convolutions approach).

All the images and masks can be found in : download_link

Files Description
aux_functions - Auxiliar functions used in processing and evaluation.

ManageFilesMulti - Prepares files and turns them in to tensors with the respective labels ready for DL

loss_functions - Some loss functions & metrics to use in training.

train - Trains a model, saving the final and best model using checkpoints.

evaluation - Evaluate models created using train

aux_functions - Creates a bar tool to download

models - Models UNET used to train 2D and 3D approachs