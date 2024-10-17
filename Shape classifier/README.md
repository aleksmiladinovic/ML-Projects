# Shape classifier

Shape classifier contains an algorithm that classifies a shape of a given image using a convolutive neural network. The input consists of images, each one containing a picture of a shape most similar to one of the three basic shapes: **rectangle**, **circle** (or **ellipse**) and a **triangle**. This data is stored in multiple files: the images are stored in files **'data[number].aca'**, while the corresponding shapes are stored in files **'data description[number].aca'**. The data files are stored with the number of pictures followed by their size, after which the images are stored as arrays.

The file **'data_generator.py'** contains a code used for generating pictures of shapes later used for training and evaluation of our neural network. Due to technical reasons the code generates a file of only 5000 pictures, hence when reading the data we open multiple files. A sample of generated pictures is provided in the folder **'sample pictures'**.

The file **'shape_classifier.py'** contains the code where we train and evaluate our neural network. The model is stored in the file **'model.pth'** while the end result is written in the file **'Output.txt'**.
