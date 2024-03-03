# Gender Identification using Transfer Learning

This repository contains code for building a gender identification model using transfer learning with the CelebA dataset. The model is built using the VGG16 architecture pre-trained on the ImageNet dataset.

## CelebA Dataset
The CelebA dataset is a popular dataset used in computer vision and deep learning for face detection and attribute recognition tasks. It consists of 202,599 face images of various celebrities with 40 binary attribute annotations per image. The dataset covers large pose variations, background clutter, and diverse people, making it suitable for training and testing models for various facial attribute recognition tasks.

- Number of images: 202,599
- Number of unique identities: 10,177
- Attribute annotations: 40 binary attributes per image

You can obtain the CelebA dataset from [Kaggle](https://www.kaggle.com/jessicali9530/celeba-dataset).

## Model Building

### 1. Model Initialization
The VGG16 model is initialized without the dense layers, and the input shape is set to (178, 218, 3).

### 2. Model Modification
- The last layer of predictions from the VGG16 model is removed.
- A new output layer with a sigmoid activation function is added to perform binary classification for gender identification.

### 3. Model Compilation
The model is compiled using the Adam optimizer and binary cross-entropy loss function. Accuracy metrics are computed during training.

### 4. Data Preparation
- Training and validation data generators are created using `ImageDataGenerator` from Keras.
- Data augmentation techniques such as shear range, zoom range, and horizontal flip are applied to the training data generator.
- Images are rescaled by 1/255.
- Training and validation data are prepared with a target size of (224, 224) and a batch size of 32.

### 5. Model Training
- The model is trained using the training dataset.
- Early stopping criteria are implemented to monitor the loss value on the validation set and stop training if the loss value doesn't improve for two consecutive epochs.
- The best model based on validation set performance is automatically saved.

## Usage
- Clone the repository.
- Download the CelebA dataset from the provided link and place it in the appropriate directory.
- Run the notebook `7_GenderIDTransfLearn_v3.ipynb` in Google Colab or any Jupyter environment.
- Train the model and evaluate its performance.
- Use the trained model for gender identification on new images.

### Sample Usage
```python
# Load the trained model
from keras.models import load_model
model = load_model('genderidtransferlearn.h5')

# Load and preprocess the sample image
from keras.preprocessing import image
import numpy as np

img_path = 'sample_image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.  # Rescale the pixel values to [0, 1]

# Perform gender identification
result = model.predict(img_array)

# Map the prediction to gender label
if result[0][0] > 0.5:
    prediction = 'Male'
else:
    prediction = 'Female'

print(f"The predicted gender is: {prediction}")
```

## Acknowledgment
The CelebA dataset was originally collected by researchers at MMLAB, The Chinese University of Hong Kong.
