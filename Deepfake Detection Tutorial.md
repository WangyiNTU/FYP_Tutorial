# **Deepfake Detection Tutorial**

### For vision-based forgery and deepfake detection, there are several Python packages and libraries you can use. Here are some commonly used ones, along with a short tutorial to get started.

## **1 Deep Learning-Pytorch**
### There are several official guidelines and resources provided by reputable organizations and institutions that can help you understand Pytorch. Here are some recommended resources:
### **[1.1 Python Basics](<https://github.com/LukeDitria/pytorch_tutorials/tree/main/section00_python_basics>)**
### **[1.2 Machine Learning Basics](<https://github.com/LukeDitria/pytorch_tutorials/tree/main/section01_numpy_ml>)**
### **[1.3 Pytorch Intro and Basics](<https://github.com/LukeDitria/pytorch_tutorials/tree/main/section02_pytorch_basics>)**
### **[1.4 Convolutions and CNNs](<https://github.com/LukeDitria/pytorch_tutorials/tree/main/section04_pytorch_cnn>)**
### **[1.5 Pytorch Tools and Training Techniques](<https://github.com/LukeDitria/pytorch_tutorials/tree/main/section06_pretraining_augmentations>)**
### **[1.6 Image Generation](<https://github.com/LukeDitria/pytorch_tutorials/tree/main/section09_generation>)**
### **[1.7 Attention](<https://github.com/LukeDitria/pytorch_tutorials/tree/main/section13_attention>)**
### **[1.8 Transformer](<https://github.com/LukeDitria/pytorch_tutorials/tree/main/section14_transformers>)**
### **[1.9 Deploying Models](<https://github.com/LukeDitria/pytorch_tutorials/tree/main/section15_deploying_models>)**

##  **2 Deepfake Detection**
### Deepfake detection on videos is a complex task that involves multiple steps, including dataset selection, model selection, training, and evaluation. The following is a structured approach to solve this problem:

### **2.1 Define the Problem**
   #### **2.1.1 Deepfake**: Deepfake is a technique that generates or modifies multimedia content by training deep neural networks to create highly realistic fake videos, audio, or images. Common methods include the following:
   - #### **FaceSwap**. FaceSwap is a graph-based method for transferring facial regions from a source video to a target video.

   - #### **Face2Face**. Face2Face is a facial reconstruction system that transfers the expression of a source video to a target video while maintaining the identity of the target person.

   - #### **Audio-visual Manipulation**. Tampering with the key words of the target person's words in the video and modifying the lip shape.

   #### **2.1.2 Deepfake Detection**: 

   - #### **Frame Consistency Analysis**. Detects whether there are signs of forgery by analyzing the continuity and consistency between video frames.

   - #### **Lighting and Shadow Analysis**. Detects inconsistencies in lighting and shadows in the video, as forged videos may have defects in these aspects.

   - #### **Motion Trajectory Analysis**. Identifies unnatural motion patterns by analyzing the movement trajectory of objects or people in the video.

   - #### **Audio-video synchronization analysis**. Detects whether the audio and video are synchronized, as forged videos may have problems with synchronization.
### **2.2 Datasets**
#### **2.2.1 Video Deepfake Dataset**: 
- #### **[Face Forensic++](<https://github.com/ondyari/FaceForensics>)**: FaceForensics++ is a forensics dataset consisting of 1000 original video sequences that have been manipulated with four automated face manipulation methods
- #### **[Celeb-DFv2](<https://github.com/yuezunli/celeb-deepfakeforensics>)**: To date, Celeb-DF includes 590 original videos collected from YouTube with subjects of different ages, ethic groups and genders, and 5639 corresponding DeepFake videos.
- #### **[TVIL](<https://github.com/ymhzyj/UMMAFormer>)**: A novel temporal video inpainting localization (TVIL) dataset that is specifically tailored for video inpainting scenes.
#### **2.2.2 Video-Audio Deepfake Dataset**:
- #### **[DFDC](<https://ai.meta.com/datasets/dfdc/>)**: The DFDC dataset is by far the largest currently publicly-available face swap video dataset, with over 100,000 total clips sourced from 3,426 paid actors, produced with several Deepfake, GAN-based, and non-learned methods.
- #### **[LAV-DF](<https://github.com/ControlNet/LAV-DF>)**: A content-driven audio-visual deepfake dataset, termed Localized Audio Visual DeepFake, explicitly designed for the task of learning temporal forgery localization. 
- #### **[FakeAVCeleb](<https://github.com/DASH-Lab/FakeAVCeleb>)**: A novel Audio-Video Multimodal Deepfake Detection dataset, which contains not only deepfake videos but also respective synthesized cloned audios.
- #### **[Deepfake1MData](<https://github.com/ControlNet/AV-Deepfake1M>)**: The dataset contains content-driven (i) video manipulations, (ii) audio manipulations, and (iii) audio-visual manipulations for more than 2K subjects resulting in a total of more than 1M videos.

### **2.3 Preprocessing**
#### **2.3.1 Extract Frames** 
#### Video framing refers to extracting individual frames (i.e. static images) from a video, which is very useful in some deepfake detection methods.
#### **2.3.2 Extract Video Features** 
#### Video feature extraction refers to extracting useful information from video data for further analysis and processing. The extracted information includes spatial features (RGB color space, texture features, edge features) and temporal features (flow).Commonly used feature extraction tools include I3D（https://github.com/piergiaj/pytorch-i3d）, etc.

### **3 Model Selection (Need to Choose One to Reproduce)**
#### **3.1 Video-base Deepfake Detection Baseline**
- #### [Thumbnail Layout for Deepfake Video Detection(TALL4Deepfake)](<https://github.com/rainy-xu/TALL4Deepfake?tab=readme-ov-file>)
#### **3.2 Multi-Model Deepfake Detection Baseline**
- #### [UMMAFormer: A Universal Multimodal-adaptive Transformer Framework for Temporal Forgery Localization](<https://github.com/ymhzyj/UMMAFormer>)
#### To help you understand better, we have provided a basic model for you to study.
- #### Attention Is All You Need: [Transformer](https://arxiv.org/abs/1706.03762)
- #### Hierarchical Vision Transformer using Shifted Windows: [Swin Transformer](https://arxiv.org/abs/2103.14030)

### **4 Model Training**
#### **4.1 Frameworks**
#### Use frameworks like TensorFlow or PyTorch to build and train your models.
#### **4.2 Loss Functions** 
#### Choose appropriate loss functions for deepfake detection tasks (e.g., cross-entropy loss for classification, mean squared error).
#### **4.3 Training Process** 
#### Split your dataset into training, validation, and test sets. Train your model using the training set and validate it using the validation set.

### **5 Evaluation**
#### Use metrics such as Average Precision and AUC (Area Under Curve) to evaluate the detection effect of the model

### **6 Implemention**
#### **6.1 Getting Started**
#### You can choose any one of the projects to start with, [TALL4Deepfake](<https://github.com/rainy-xu/TALL4Deepfake?tab=readme-ov-file>) or [UMMAFormer](<https://github.com/ymhzyj/UMMAFormer>).
#### **6.2 Environment** 
- #### Create a new conda environment and install dependencies.
- #### Sometimes you may encounter errors, package conflicts, and other issues. Please be patient and use Google to search for solutions.
#### **6.3 Data Preparation**
- #### Choose one dataset,  [FaceForensic++ (For TALL4Deepfake)](<https://github.com/ondyari/FaceForensics>) or [LAV-DF (For UMMAFormer)](<https://github.com/ControlNet/LAV-DF>).
- #### The second option is to build a training set that is uniquely tailored to you.
#### **6.4 Training**
#### Run the following script (an example) to train your model from scratch.
```python
 python train.py
```
#### If you don't want to train your network from scratch, you can download pre-trained model checkpoints.
#### **6.5 Evaluation** 
#### Run the following script (an example) to evaluate your model.
```python
 python evalate.py
```
#### **6.6 Testing**
#### Run the following script (an example) to test your model.
```python
 python test.py
```

### **7 Deployment**
#### Can be deployed in real-time applications (such as anti-fraud platforms), paying attention to optimizing program processing speed.

### **8 Reference**
#### **8.1 Video-base Forgery and Deepfake Detection**
#### Thumbnail Layout for Deepfake Video Detection
#### Paper: https://arxiv.org/pdf/2403.10261
#### **8.2 Multi-Model Forgery and Deepfake Detection**
#### UMMAFormer: A Universal Multimodal-adaptive Transformer Framework for Temporal Forgery Localization
#### Paper: https://arxiv.org/abs/2308.14395


