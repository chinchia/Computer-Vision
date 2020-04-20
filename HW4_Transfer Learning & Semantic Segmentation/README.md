## Question 1. Transfer Learning in CNN

### Implementation
1. Backbone: Alexnet.
2. Quantitative metric: Top-1 accuracy.
3. Achieve validation accuracy higher than 89.5% using VGG-16 as pretrained model.

## Question 2. Semantic Segmentation

### Implementation
1. Eliminate the skip-connection so connection so the output of convolution layers of FCN8s will be directly upsampled for 32x.
2. Reduce the number of classes from 11 to 3.
