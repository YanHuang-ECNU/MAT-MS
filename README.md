# Gap-free MODIS NDSI construction
## Getting Started
### Prerequisites
1. NVIDIA GPU + CUDA<br>
2. Python 3<br>
3. Torch >= 2.0.1
### Dataset
1. Cloud mask dataset<br>
2. Train dataset with four channels, which are true NDSI, 
spatiotemporal interpolated NDSI, elevation and temperature<br>
3. Test and Strict_Test dataset, which is same as Train dataset
## Training
Run the $train$ function.
You can specify various parameters of the model, 
which are defined in lines 16-23 of model.py.
## Testing
Run the $test$ function.
You can specify which epoch and batch of model to use as test parameters.
## Acknowledgments
The code for the model draws inspiration from 
[Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
and
[Generative Image Inpainting](https://github.com/JiahuiYu/generative_inpainting)