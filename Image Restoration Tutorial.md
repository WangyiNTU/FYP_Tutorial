# Image Restoration Related Vision Overall Guidance

For the task of image restoration, there are several Python packages and libraries that you can use. Here are some popular ones along with brief tutorials on how to get started.

## **1 Deep Learning-Pytorch**
There are several official guidelines and resources provided by reputable organizations and institutions that can help you understand Pytorch. Here are some recommended resources:
### **[1.1 Python Basics](<https://github.com/LukeDitria/pytorch_tutorials/tree/main/section00_python_basics>)**
### **[1.2 Pytorch Intro and Basics](<https://github.com/LukeDitria/pytorch_tutorials/tree/main/section02_pytorch_basics>)**
### **[1.3 Convolutions and CNNs](<https://github.com/LukeDitria/pytorch_tutorials/tree/main/section04_pytorch_cnn>)**
### **[1.4 Pytorch Tools and Training Techniques](<https://github.com/LukeDitria/pytorch_tutorials/tree/main/section06_pretraining_augmentations>)**
### **[1.5 Image Generation](<https://github.com/LukeDitria/pytorch_tutorials/tree/main/section09_generation>)**
### **[1.6 Trained Model Interpretation](<https://github.com/LukeDitria/pytorch_tutorials/tree/main/section10_interpretation>)**
### **[1.7 Attention](<https://github.com/LukeDitria/pytorch_tutorials/tree/main/section13_attention>)**
### **[1.8 Transformer](<https://github.com/LukeDitria/pytorch_tutorials/tree/main/section14_transformers>)**
### **[1.9 Deploying Models](<https://github.com/LukeDitria/pytorch_tutorials/tree/main/section15_deploying_models>)**

## Image restoration
The image restoration task aims to restore an accurate, precise, and complete original image from a damaged or distorted image. According to different types of damaged images, restoration tasks can be divided into different tasks, such as image denoising, image dehazing, image inpainting, image super-resolution and so on.

### 1. **Principles and Challenges**
The basic principle of image inpainting is to use the inherent redundancy of the image and the relationships between neighboring pixels to restore the pixel values of damaged areas. However, image inpainting faces the following challenges:
   - **Context Understanding of Missing Areas**: The model needs to accurately understand the context of the image to generate natural and coherent content.
   - **Detail Restoration**: Restoring missing areas requires attention to detail (such as texture and edges), especially in high-resolution images.
   - **Diversity of Training Data**: The diversity and quality of the training dataset impact the model's generalization ability, ensuring it can handle various types of missing areas.

Addressing these challenges requires continuous improvement in model architecture, training strategies, and loss functions.

We start with the specific task of image dehazing to get acquainted with image restoration techniques.
The single image dehazing aims to recover the clean scene from
the corresponding hazy image.
### 2. **Data Collection**
   - **Datasets**: Use existing datasets or create your own. Some popular datasets include:
     - **Dehazing Datasets**: 
       - [ [RESIDE]](https://sites.google.com/view/reside-dehaze-datasets/reside-v0): RESIDE highlights diverse data sources and image contents, and is divided into five subsets, each serving different training or evaluation purposes.
       - [[HAZE4K]](https://github.com/liuye123321/DMT-Net): Haze4k is a synthesized dataset with 4,000 hazy images, in which each hazy image has the associate ground truths of a latent clean image, a transmission map, and an atmospheric light ma.

### 3. **Preprocessing**
   - **Data Augmentation**: Apply techniques such as rotation, scaling, and flipping to increase the diversity of your training data.Here is a popular augmentation library:[Albumentations documentation](https://albumentations.ai/docs/)
   - **Normalization**: Normalize images to ensure consistent input to the model.

### 4. **Model Selection**
   - **A CNN-based U-shape architecture**: [DEA-net](https://github.com/cecret3350/DEA-Net).
   - **Advanced algorithm**: A transformer-based model [DehazeFormer](https://github.com/IDKiro/DehazeFormer).

To help you understand better, we have provided a basic model for you to study.

  - **A CNN-based model**: [U-net](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28)
- **Attention Is All You Need**: [Transformer](https://arxiv.org/abs/1706.03762)


### 5. **Model Training**
   - **Frameworks**: Use frameworks like TensorFlow or PyTorch to build and train your models.
   - **Loss Functions**: Choose a appropriate loss function to evaluate the performance of the model. (e.g. L1 loss function (i.e., mean absolute error) ).
   - **Training Process**: Split your dataset into training, validation, and test sets. Train your model using the training set and validate it using the validation set.

### 6. **Evaluation**
   - **Metrics**: Peak signal noise ratio (PSNR) and structural similarity index (SSIM) are commonly used indicators in restoration tasks to measure the performance of models.
   - **Visualization**: Visualize the results to understand how well the model is performing in terms of hand and object detection and pose estimation.

## Implemention

###  Getting Started
You can choose any one of the projects to start with, [DEA-Net](https://github.com/cecret3350/DEA-Net) or [DehazeFormer](https://github.com/IDKiro/DehazeFormer).

Create a new conda environment and install dependencies


### 1. Environment
Create a new conda environment and install dependencies.

Sometimes you may encounter errors, package conflicts, and other issues. Please be patient and use Google to search for solutions

**Tutorial Steps:**
- Install mediapipe.
- Use the hand tracking solution provided by Mediapipe to detect hands in images or video streams.
- Check the doc for hand detection and pose estimation examples and usage: [mediapipe documentation for hands landmark](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python)

### 2. Data Preparation

1. Choose one dataset,  [RESIDE] or [HAZE4K].
2. The second option is to build a training set that is uniquely tailored to you.

### 3. Training

Run the following script (an example) to train your model from scratch.

```python
 python train.py
```

If you don't want to train your network from scratch, you can download pre-trained model checkpoints.


### 4. Evaluation

Run the following script (an example) to evaluate your model.
```python
 python evalate.py
```
### 5. Testing

Run the following script (an example) to test your model.
```python
 python test.py
```

## Reference
### For more details, you can refer to the following links.

### [DEA-Net: Single image dehazing based on detail-enhanced convolution and content-guided attention] 
paper: https://arxiv.org/abs/2301.04805

github: https://github.com/cecret3350/DEA-Net?tab=readme-ov-file

### [Vision Transformers for Single Image Dehazing]
paper: https://arxiv.org/abs/2204.03883

github: https://arxiv.org/abs/2204.03883
