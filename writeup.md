# **Behavioral Cloning**

## Writeup
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)
[image1]: ./examples/image_preprocessing.png "Image preprocessing example"
[image2]: ./examples/data_distribution.png "Distribution of steering angle in the data"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup.md summarizing the results (this file)
* full_speed_test.mp4 visualizing the performance of the model using side cameras
* Another set of files in the nice20 directory with same as above but for the model not using the side cameras

#### 2. Submission icludes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```
I have included 2 models simpler:
- one that does not use the side cameras (nice20)
- a model that uses side cameras and performs a lot smoother (main)


#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I based my model on the Nvidia model discussed in chapter 14 of the Behavioral Cloning project.
The input was initially set to match the size of the images captured by the simulator.
Later on I decided to resize the images to 64x64 pixels. That has dramatically reduced the memory footprint used by the data and I was able to train the model on more samples. It also has significantly brought the training time down.

The data is normalized in the model using a Keras lambda layer (code line 50).

#### 2. Attempts to reduce overfitting in the model

Inspired by one of the cs321n lectures I have decided to use batch normalization as it takes care of the initialization and extra regularization. It has reduced the number of epochs necessary to train and it is preventing the vanishing gradient problem.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 81). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an Adam optimizer. In the beginning I did override the learning rate to 0.1 and decay 0.1 to speed up the initial training. But with keras 2 these arguments are ignored and Adam optimizer is good as it is anyway.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. The base was the training data provided by Udacity.  I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach
This submission is actually my second model. 
My computer crashed 4 days ago and I lost all my work with my first model (which performed superbly at full speed on track 1 and made it through half of track 2 on just track 1 training data). 
I was forced to start from scratch yesterday to create a substitute in a hurry. I shall describe the design approach for both models here.

The overall strategy for deriving a model architecture was to use the best model and components I could find and build on the experience of others.

That is why right from beginning I used the Nvidia model (https://arxiv.org/abs/1604.07316) enhanced with batch normalization layers. 

The model guaranteed enough capacity for the task and the batch normalization layers made sure that the network gets used to the full capacity by effectively eliminating dead neurons.

The first model I built was using the side cameras. The hardest part for me was to find the right offsets to use for steering for the side cameras. More details below.

#### 2. Final Model Architecture

The final model used the same architecture but I got frustrated with looking for the right camera angles and decided to just use the main camera and collect recovery data.

Detailed model architecture:

|Layer (type)                 |Output Shape        |Param #   |
|-----------------------------|:------------------:|---------:|
|cropping2d_1 (Cropping2D)    |(27, 64, 3)         |0         |
|lambda_1 (Lambda)            |(27, 64, 3)         |0         |
|conv2d_1 (Conv2D)            |(14, 32, 24)        |1824      |
|batch_normalization_1 (Batch |(14, 32, 24)        |96        |
|activation_1 (Activation)    |(14, 32, 24)        |0         |
|conv2d_2 (Conv2D)            |(7, 16, 36)         |21636     |
|batch_normalization_2 (Batch |(7, 16, 36)         |144       |
|activation_2 (Activation)    |(7, 16, 36)         |0         |
|conv2d_3 (Conv2D)            |(4, 8, 48)          |43248     |
|batch_normalization_3 (Batch |(4, 8, 48)          |192       |
|activation_3 (Activation)    |(4, 8, 48)          |0         |
|conv2d_4 (Conv2D)            |(4, 8, 64)          |27712     |
|batch_normalization_4 (Batch |(4, 8, 64)          |256       |
|activation_4 (Activation)    |(4, 8, 64)          |0         |
|conv2d_5 (Conv2D)            |(2, 6, 64)          |36928     |
|batch_normalization_5 (Batch |(2, 6, 64)          |256       |
|activation_5 (Activation)    |(2, 6, 64)          |0         |
|flatten_1 (Flatten)          |(768)               |0         |
|dropout_1 (Dropout)          |(768)               |0         |
|dense_1 (Dense)              |(100)               |76900     |
|batch_normalization_6 (Batch |(100)               |400       |
|activation_6 (Activation)    |(100)               |0         |
|dense_2 (Dense)              |(50)                |5050      |
|batch_normalization_7 (Batch |(50)                |200       |
|activation_7 (Activation)    |(50)                |0         |
|dense_3 (Dense)              |(10)                |510       |
|batch_normalization_8 (Batch |(10)                |40        |
|activation_8 (Activation)    |(10)                |0         |
|dense_4 (Dense)              |(1)                 |11        |



#### 3. Data preprocessing
Initially I was working with the full size images.
I was cropping them to get rid of the top part of the image filled with the sky and the bottom part with the car's hood.
When I got the first model working perfectly with the first track I tested it on track 2 and it made it to the first tree where it casted a shadow on the road. 
To combat that I used adaptive histogram equalization (CLAHE) on the image and the model was able to make it all the way to the top of the moutains. There it losts it's marbles because of the winding road below on the right.
However using it decreased performance on the first track.

I was working on getting the model to perform on both tracks when my computer crashed.

The second model does not use CLAHE as I was trying to get something that would make me pass the deadline quick.

I'm adding the mirror images with reversed steering values in order to prevent the model of being biased towards turning to the left.

The model that uses side cameras adds mirror images of both cameras as well. That results with a lot of extra samples.

As I was training the models I was trying to train on more and more data until it wouldn't fit in my memory. That's when I tried resizing to 64x64 after making sure it did not impede the performance.

I also experimented by converting to HSV as my mentor suggested and to YUV as the Nvidia paper suggested but I didn't see any significant improvements so decided against it and kept things simple.

I was also thinking about nulling out the top corners but decided against it as in real life that would not be applicable as we might react on incoming objects in that space.

Example of the source image and preprocessed one:

![alt text][image1]

Distribution of the steering angles in the data set:

![alt text][image2]

As you can see the data is hevily biased towards turning left. This is countered by adding mirror image to the pipline as well as described above.

#### 4. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using top speed center lane driving.

I then recorded one lap of slow driving (even slower on the turns) to get more frames for a given pass. I was trying to Slowing down to minimum speed at turns 

Then I recorded few laps driving the other way. This was to train the model on more diverse input so it could generalize better.

For the first model I did not record any extra recovery data other than the few misshaps during my regular driving.
The idea behind it was to try to train the model to make smooth small corrections to prevent getting into trouble to begin with.

To augment the data, I flipped the center images and also used the left and right camera images with corrected steering angle. Figuring out the optimum correction value took me a while as the suggested starting value was way too high (0.2) and I thought that it needed just a minor tweak. It turned out that it was the main reason that the model was over correcting and was really swiveling from side to side too much.  I ended up using 0.04 as it gave nice small corrections, but big enough that the car could make it through the turns.

Next I tried to make the model be more willing to turn by dropping training examples with 0 steering angle with certain probability. I've used values between 0.3 and 0.7 with success. Lower values made the car go smoother, higher values made the car stay closer to the center when making turns.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs varied depending on the data. I used 11 for smaller training sets and 4-6 on the larger ones. That number was determined by looking on the drop of validation loss with each epoch. Sometimes it would plateau at 0.0150 and and sometimes at 0.0050. 

#### 5. Observations
#### Low validation loss does not always mean that the car would stay on track. 
I would get as low as 0.0032 on 30% validation split and the car would just go in circles.

#### Initialization matters
It's worth running and testing the model multiple times before making changes. 
That is caused by hitting local optima during training.
Many times the model would train to straddle the side lines instead of the center of the road.

#### More data is not always the answer

I recorded a lot of laps using my joystick. They were all smooth. Ultimately however I used just the udacity run with a sinlgle slow run. Adding more data caused the model to act hectic. 
This is very surprising.

#### 6. Enhancements
I really wanted to get it to work with track 2. 
I trained the model with track 2 data and I made it stick to a single lane. But it never made it through the entire track.
I've noticed that because of the steepness of the road it might be beneficial to crop it more from the top to get rid of the distant winding road.

#### 7. Surprise

As I was writing this I was working on rewriting the original model that uses the side cameras. And Just as I was wrapping point 6 it worked. I changed the following:

 - Added a dropout before the fully connected layers
 - Changed the side camera steering offset to 0.07
 - Trained for 7 epochs
 - Used only udacity data and data from the single low speed pass.
 - Removing %30 of samples with 0 steering value to prevent the model from going straight on the turns

I'm approaching the deadline and I'm in a hurry so please forgive me for the lack of visualizations here.
