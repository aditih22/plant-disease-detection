# Plant-Disease-Detection

**Problem Statement:**

In this project, we successfully developed a Convolutional Neural Network (CNN) to predict plant diseases based on images. The process began by loading the dataset into Google Colab via Google Drive, followed by visualizing and normalizing the images. Normalization is a critical step as it ensures that the data is scaled appropriately, improving the model's performance.

Once the dataset was prepared, we designed and trained a CNN model specifically for plant disease detection. The model was then tested using images of plants to determine its ability to correctly identify diseased plants. The results were promising, demonstrating the model's effectiveness in classifying plant diseases.

This model holds significant potential for practical applications in agriculture. It can be utilized by agricultural firms and farmers to quickly and accurately diagnose plant diseases, enabling them to take timely action. By doing so, they can prevent the spread of diseases, reduce crop loss, and ultimately increase agricultural yield. The ability to predict plant diseases early helps in minimizing waste and maximizing the productivity of crops, making this model a valuable tool for modern agriculture.



This code snippet defines a Convolutional Neural Network (CNN) using Keras' `Sequential` API, which is used for building simple, layer-by-layer neural networks. The model is designed to classify images into one of three categories, which could correspond to different plant diseases or healthy plants. Here's a detailed explanation of each part:

### 1. **Model Initialization**
   ```python
   model = Sequential()
   ```
   - The `Sequential()` model is a linear stack of layers, meaning each layer has exactly one input tensor and one output tensor.

### 2. **First Convolutional Layer**
   ```python
   model.add(Conv2D(32, (3, 3), padding="same", input_shape=(256,256,3), activation="relu"))
   ```
   - **Conv2D Layer:**
     - **Filters:** 32 filters (also known as kernels) are used to scan the image.
     - **Kernel Size:** The size of each filter is 3x3 pixels.
     - **Padding:** `"same"` padding ensures that the output has the same width and height as the input by adding zeros around the border if necessary.
     - **Input Shape:** `(256, 256, 3)` specifies the input image dimensions: 256x256 pixels with 3 channels (RGB).
     - **Activation Function:** `ReLU` (Rectified Linear Unit) introduces non-linearity, helping the network to learn complex patterns.

### 3. **First Max Pooling Layer**
   ```python
   model.add(MaxPooling2D(pool_size=(3, 3)))
   ```
   - **MaxPooling2D Layer:**
     - **Pool Size:** 3x3. This layer reduces the spatial dimensions of the feature maps (width and height) by taking the maximum value in each 3x3 block.
     - **Purpose:** It reduces the computational complexity and helps in extracting the most prominent features.

### 4. **Second Convolutional Layer**
   ```python
   model.add(Conv2D(16, (3, 3), padding="same", activation="relu"))
   ```
   - **Conv2D Layer:**
     - **Filters:** 16 filters are used in this layer, reducing the number of filters compared to the first convolutional layer.
     - **Kernel Size:** 3x3, the same as the first convolutional layer.
     - **Padding:** `"same"` padding ensures the output dimensions match the input.
     - **Activation Function:** `ReLU` is again used to introduce non-linearity.

### 5. **Second Max Pooling Layer**
   ```python
   model.add(MaxPooling2D(pool_size=(2, 2)))
   ```
   - **MaxPooling2D Layer:**
     - **Pool Size:** 2x2. This further reduces the spatial dimensions of the feature maps, helping to compress the features learned by the convolutional layers.

### 6. **Flattening Layer**
   ```python
   model.add(Flatten())
   ```
   - The `Flatten()` layer converts the 2D feature maps from the previous layer into a 1D vector. This vector serves as the input for the fully connected (dense) layers that follow.

### 7. **First Dense Layer**
   ```python
   model.add(Dense(8, activation="relu"))
   ```
   - **Dense Layer:**
     - **Units:** 8 neurons in this layer. Each neuron is fully connected to all the neurons in the previous layer.
     - **Activation Function:** `ReLU` is used for non-linearity, enabling the model to learn complex relationships.

### 8. **Output Layer**
   ```python
   model.add(Dense(3, activation="softmax"))
   ```
   - **Dense Layer:**
     - **Units:** 3 neurons, corresponding to the 3 classes (e.g., 3 different plant conditions).
     - **Activation Function:** `Softmax` is used to output a probability distribution across the 3 classes, with the sum of the probabilities equal to 1. The class with the highest probability is chosen as the model's prediction.

### 9. **Model Summary**
   ```python
   model.summary()
   ```
   - This command outputs a summary of the model architecture, showing each layer's type, output shape, and the number of parameters that need to be learned during training.


### Process:
#### Step 1 : Loading The Dataset

![image](https://github.com/user-attachments/assets/4f1ef7db-61c5-4e6b-8919-1bf8ad072db9)

#### Step 2 : Building The Model

![image](https://github.com/user-attachments/assets/092445b0-52a9-4522-b0ed-9a02d1b7b72c)

#### Step 3 : Training The Model

![image](https://github.com/user-attachments/assets/a5c252cb-902a-4462-b320-f17d02da2e79)

#### Step 4 : Displaying The Model Accuracy

![image](https://github.com/user-attachments/assets/3f819bcf-f5e3-47b1-bb57-9f056014a00f)

#### Step 5 : Testing the model with Image

![image](https://github.com/user-attachments/assets/f10e5915-549b-45a3-8819-a4887625879e)

#### Step 6 : Lauching The App on streamlit 

![image](https://github.com/user-attachments/assets/79b8cdf8-7872-4803-b414-4adc4d5e4b77)
