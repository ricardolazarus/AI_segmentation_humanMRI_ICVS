Deep Learning to segment different brain tissues In this project, the main objective is to segment different brain tissues both in Humans (with a 2D approach). 

Files Description: The objective of each file is:

NumpyImage - Pre-processing steps ObjectModel.py - (In some folders this name maybe different). Creates the models architectures

ManageFiles.py - Prepares files and turns them in to tensors with the respective labbles ready for DL

loss_functions.py - Some loss functions created to see some metrics.

train.py - It calls other functions. Creates the model of Objectmodel.py, Prepares the files with PrepareFiles.py, and starts training. Compiles, defines epochs, loss functions, optmizers etc. And saves the final and best model of modelcheckpoint.

writegraphs.py - Used to create the tensorboard graphs with validation and training lines in the same graph

helpingtools.py - Creates a bar tool to download

visualize.py - To save feature maps. Only used in the final after training, otherwise it would take a long time
