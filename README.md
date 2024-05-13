# interactive behavior model

## model

VanillaAE -- trained via reconstruction. The stride and size of convolution is length of joints in layer 1. Didn't show significant gradient decent during training.
VanillaAEv2 -- trained via reconstruction. The stride and size of convolution is 3 and 2 in each layer. Showed gradient decent during training.

#VanillaAEPred version_0 trained with all normal mice and test on abnormal mice in prediction framework.
#VanillaAEPred version_1 trained with half of normal mice and validiate on another half of noraml mice and test on abnormal mice. The data used are train, test and val.
VanillaAEPred version_0 trained with all normal mice and test on abnormal mice. The data used are train1, test1 and val1.

## data

train, test and val are the data that filter out joints of other mice fall out of social distance, without care other joints on the same mice.
train1, test1 and val1 are the data that filter out whole joints of one mouse if all of their joints fall out of social distance.

