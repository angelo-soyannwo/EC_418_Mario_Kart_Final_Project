# Turn Clasifier

The road_classifier and gray_scale_road_classifier modules train neural networks to detect sharp turns within frames in the game. These predictions take the forms of labels 0, 1, and 2 (denoting a sharp left, straight path, or sharp right respectively).

## Training models

The following two commands can be used in the terminal to train the road_classifier and the gray_scale_road_classifier neural networks respectively.

$python3 train.py -n [num_epochs]

$python3 train_gray_scale.py -n [num_epochs]

The -c flag can be used to continue training the models for all of the above training commands.
