# EC_418_Mario_Kart_Final_Project

My EC 418 Reinforcement Learning final project. A neural network based agent that is capable of playing mario kart on the PySuperTuxKArt emulator.

<p align="center">
 <a href="https://drive.google.com/file/d/1UcUQiL_LJX4ywOeCUIq87zp5nzuJVGvW/view?usp=drive_link" target="_blank">Example Video</a>
</p>

<p align="center">
  <img width="460" height="300" src="https://github.com/angelo-soyannwo/EC_418_Mario_Kart_Final_Project/blob/main/mario_kart.png">
</p>

The agent in question is the planner neural network to predict the aim point of each frame within the game. The green circle in the above image is the predicted aim point of the planner model, while the red circle is the actual aim point. GrayScaleRoadClassifier.nn and RoadClassifier.nn produce predictions for wheter a sharp turn is in the image. Editing the planner.py file, the utils.py file and the controller.py file will enable you to alter the agent accordingly if you would like.

## Installation Info

install anaconda

then execute: 

$pip install -U PySuperTuxKart

$conda install -c anaconda mesa-libegl-cos6-x86_64

## How to Generate Training Images

$python3 -m utils zengarden lighthouse hacienda snowtuxpeak cornfield_crossing scotland

The above command should generate a folder filled with images and their corresponding aim points.

## Executing the code

$python3 -m planner [TRACK_NAME] -v

TRACK_NAMES = zengarden, lighthouse, hacienda, snowtuxpeak, cornfield_crossing, scotland

The above code should cause the emulator to start. It will print 0, 1, or 2 in the terminal when running (denoting a sharp left, straight path, or sharp right respectively).

## Training models

$python3 train.py -n [num_epochs]

The above function trains the planner module.

In the turn_classifier folder, the following two commands can be used in the terminal to train the road_classifier and the gray_scale_road_classifier neural networks respectively.

$python3 train.py -n [num_epochs]

$python3 train_gray_scale.py -n [num_epochs]

The -c flag can be used to continue training the models for all of the above training commands.
