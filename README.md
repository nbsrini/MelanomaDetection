# Project Name
> Melanoma Detection Assignment - Convolutional Neural Networks




## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)

<!-- You can include any other section that is pertinent to your problem -->

## General Information
Problem Statement and Objective:
To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution that can evaluate images and alert dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.

Data context
The dataset consists of 2357 images of malignant and benign oncological diseases, which were formed from the International Skin Imaging Collaboration (ISIC). All images were sorted according to the classification taken with ISIC, and all subsets were divided into the same number of images, with the exception of melanomas and moles, whose images are slightly dominant.

The data set contains the following diseases, which will form classes in the output:

Actinic keratosis
Basal cell carcinoma
Dermatofibroma
Melanoma
Nevus
Pigmented benign keratosis
Seborrheic keratosis
Squamous cell carcinoma
Vascular lesion


Approach and High Level Steps:
We will learn from the base starter code provided and implement the same for our problem statement. The model training may take time to train as we will be working with large epochs, therefore we will use GPU runtime

1) Data Reading/Data Understanding → Defining the path for train and test images
2) Dataset Creation → Create train & validation dataset from the train directory with a batch size of 32. Resize the images to 180*180.
3) Dataset visualisation → Create a code to visualize one instance of all the nine classes present in the dataset
4) Model Building & training :
Create the CNN model to accurately detect 9 classes present in the dataset. Rescale images to normalize pixel values between (0,1).
Choose an appropriate optimiser and loss function for model training
Train the model for ~20 epochs
Write the findings after the model fit. Check if there is any evidence of model overfit or underfit.
5) Choose an appropriate data augmentation strategy to resolve underfitting/overfitting
6) Model Building & training on the augmented data :
Create the CNN model to accurately detect 9 classes present in the dataset. Rescale images to normalize pixel values between (0,1).
Choose an appropriate optimiser and loss function for model training
Train the model for ~20 epochs
Write your findings after the model fit, see if the earlier issue is resolved or not?
7) Class distribution: Examine the current class distribution in the training dataset - Which class has the least number of samples? - Which classes dominate the data in terms of the proportionate number of samples?
8) Handling class imbalances: Rectify class imbalances present in the training dataset with Augmentor library.
9) Model Building & training on the rectified class imbalance data :
Create the CNN model to accurately detect 9 classes present in the dataset. Rescale images to normalize pixel values between (0,1).
Choose an appropriate optimiser and loss function for model training
Train the model for ~30 epochs
Write the findings after the model fit, see if the issues are resolved or not?

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->

## Conclusions
We have developed a CNN model to detect the 9 classes with a training and validation accuracy of >86%.

The summary of model performance is evaluated as follows

-  Epoch 28 indicates the maximum performance of the model at training accuracy of  90.6% and validation accuracy of 86.5%
 - The low difference between training accuracy and validation accuracy means that the overfitting found in the previous iteration of the model has been resolved
 - The model training executed until Epoch 30 however the validation accuracy did not improve beyond Epoch 28
 - Addressing class imbalance and adding a third convolution layer have really helped with improving the overall accuracy of the model and avoid overfitting

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->


## Technologies Used
- Pandas
- Numpy
- tensorflow
- Keras
- matplotlib
- seaborn

<!-- As the libraries versions keep on changing, it is recommended to mention the version of library used in this project -->


## Contact
Created by [@nbsrini] - feel free to contact me!


<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->