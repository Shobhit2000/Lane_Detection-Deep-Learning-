# Lane_Detection(Deep-Learning)

This is a Deep Learning based Lane Detection model that detects the lane lines on the road which takes in an image and outputs the segmentation mask with lane markings indicated in white and the background as black.

It uses Resnet50 with initial pretrained weights as the encoder using transfer learning and a custom self constructed decoder with binary crossentropy as the loss function.

## Results
The top image represents the ground truth of the lane line and the corresponding prediction.

<p align="center">
  <img src="https://github.com/Shobhit2000/Lane_Detection_Deep-Learning/blob/master/images/ground_truth.PNG">
</p>

<p align="center">
  <img src="https://github.com/Shobhit2000/Lane_Detection_Deep-Learning/blob/master/images/pred.PNG">
</p>
