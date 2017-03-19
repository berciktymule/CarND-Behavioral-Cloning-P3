#**Behavioral Cloning**

##Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I based my model on the Nvidia model discussed in chapter 14 of the Behavioral Cloning project.
The input was initially set to match the size of the images captured by the simulator.
Later on I decided to resize the images to 64x64 pixels. That has dramatically reduced the memory footprint used by the data and I was able to train the model on more samples. It also has significantly brought the training time down.

The model includes ELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18).

####2. Attempts to reduce overfitting in the model

Inspired by one of the cs321n lectures I have decided to use batch normalization as it takes care of the initialization and extra regularization. It has reduced the number of epochs necessary to train and it is preventing the vanishing gradient problem.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer. I did override the learning rate to start off with a greater  so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ...

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that ...

Then I ...

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using top speed center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded one lap of slow driving (even slower on the turns) to get more frames for a given pass.

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I recorded few laps driving the other way. This was to train the model on more diverse input so it could generalize better.

I did not record any extra recovery data other than the few misshaps during my regular driving.
The idea behind it was to try to train the model to make smooth small corrections to prevent getting into trouble to begin with.

To augment the data, I flipped the center images and also used the left and right camera images with corrected steering angle. Figuring out the optimum correction value took me a while as the suggested starting value was way too high (0.2) and I thought that it needed just a minor tweak. It turned out that it was the main reason that the model was over correcting and was really swiveling from side to side too much.  I ended up using 0.04 as it gave nice small corrections, but big enough that the car could make it through the turns.

Next I tried to make the model be more willing to turn by dropping training examples with 0 steering angle with certain probability. I've used values between 0.3 and 0.7 with success. Lower values made the car go smoother, higher values made the car stay closer to the center when making turns.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
