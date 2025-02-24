# tumor_detection

## Chapter 2: Methods
### 2.1 CNN
A Convolution Neural Network (CNN) is a type of deep learning algorithm that is particularly well-suited for image recognition and processing tasks. It is made up of multiple layers, including convolution layers, pooling layers and fully connected layers.
![image](https://github.com/user-attachments/assets/e0d9af55-f58c-48fb-9d9b-cc1bc5472bdd)
a. Convolution Layer
The convolution layers are typically the first layer, abstracting the images using filters. Each filter’s response to the input image is represented by a set of features.
![image](https://github.com/user-attachments/assets/69391ba8-4836-4735-93ab-3ab7ef9d15e3)
The output of the convolutional layer can be expressed as:Vc=\left[\frac{M_C-F_C+2P}{S_C}+1\right]
b. Pooling Layer
The second layer, known as pooling, is added to the feature map to reduce size while maintaining key features. This reduction is performed with fewer parameters to prevent overfitting.
![image](https://github.com/user-attachments/assets/733c382b-9573-40d8-9375-f4e95c487098)
c. Fully Connected Layer
In fully connected layers, the neuron applies a linear transformation to the input vector through a weight matrix. A non-linear transformation is then applied to the product through a non-linear activation function f.
Equation for a non-linear transformation:     y_{jk}\left(x\right)=f\left(\sum_{i=1}^{n_H}{w_{jk}x_i+w_{j0}}\right)
![image](https://github.com/user-attachments/assets/95fc7250-8ef2-43f0-ab39-48adfc8098df)
## 2.2 ResNet101
A residual neural network is a deep learning architecture in which the layers learn residual functions with reference to the layer inputs. The residual connection stabilizes the training and convergence of deep neural networks with hundreds of layers.
![image](https://github.com/user-attachments/assets/684191a3-40a6-4d7a-83be-8a1eb60c9afc)
## 2.3 VGG-16
VGG-16 is a CNN architecture, consisting of 16 layers, including 13 convolution layers and 3 fully connected layers. VGG16 is renowned for its simplicity and effectiveness, as well as its ability to achieve strong performance on various computer vision tasks. The model’s architecture features a stack of convolution layers followed by max-pooling layers, with progressively increasing depth.
![image](https://github.com/user-attachments/assets/d2f37b72-634f-4774-a4eb-272cd1cd2e08)
## 2.4 VGG-19
VGG-19 is a deep CNN with 19 weight layers, comprising 16 convolutional layers and 3 fully connected layers. It trained on over 1.2 million images from the ImageNet database.
![image](https://github.com/user-attachments/assets/7ed5abee-58cc-46b8-8eca-82c631536494)




## Chapter 4: Proposed work
### 4.1 Experimental setup
### 4.1.1 Computer configuration
OS			: Windows / Linux
RAM			: 16 GB
HDD			: 256 GB
GPU			: 12 GB
Jupyter notebook	: v7.2
#4.2 Dataset
This dataset comprises a comprehensive collection of augmented MRI images of brain tumors, organized into two distinct folders: 'Yes' and 'No'. The 'Yes' folder contains 9,828 images of brain tumors, while the 'No' folder includes 9,546 images that do not exhibit brain tumors, resulting in a total of 19,374 images. All images are in PNG format, ensuring high-quality and consistent resolution suitable for various machine learning and medical imaging research applications.
Given the challenges associated with acquiring a large number of MRI images due to the high costs and limited availability, data augmentation techniques were employed to expand the dataset.
Dataset Structure
The dataset is organized into two main folders:
•	Yes: Contains 9,828 PNG images of brain tumors.
•	No: Contains 9,546 PNG images that do not exhibit brain tumors.
Total images: 19,374.
Data Format:
•	File Format: PNG
Dataset image resolution:
•	256x256x3
#4.3 Execution
To train the model, import the require machine learning library functions and proposed models. Set parameters like- Image size (128,128), Batch size (128), Epochs (100), Learning rate (0.0001), Random state (32). Preprocessing the data with rescale (1./255) and validation split (0.3). Split the data into Training (70%), Validation (20%), and Testing (10%). Defined models are CNN, VGG-16, VGG-19, ResNet101.After that, train the model and evaluate the score, i.e., training accuracy and loss, validation accuracy and loss, test accuracy and loss. To visually represent the outcomes, plot comparison of accuracy and loss graph of proposed models. For closer look of every trained model, plot the individual accuracy and loss graph. At last, generate the Confusion metrics and Performance metrics.
![image](https://github.com/user-attachments/assets/efa4e254-029a-4a30-a25c-698d42b592ce)

### 4.4 Output
Model		Training (%)	Validation (%)	Test (%)

CNN	Accuracy	99.33	86.71	96.01
	Loss	2.10	77.60	18.76

VGG-16	Accuracy	100	95.71	98.71
	Loss	0.12	15.33	5.39

VGG-19	Accuracy	100	95.79	98.74
	Loss	0.25	15.43	4.72

ResNet-101	Accuracy	90.84	80.28	88.58
	Loss	23.22	46.13	28.55


![image](https://github.com/user-attachments/assets/1cc5adbc-7abd-4cb4-8042-953fee5757fb)

![image](https://github.com/user-attachments/assets/d5fd3ae2-792e-4a80-9b2c-cc587698b819)

![image](https://github.com/user-attachments/assets/1a70b074-8f0d-4638-a998-b2d3c291ba1c)

![image](https://github.com/user-attachments/assets/18385168-a01e-458a-8aa5-6e58da641e2d)

![image](https://github.com/user-attachments/assets/ff074375-ceac-4223-b314-0606cdadf4cc)

![image](https://github.com/user-attachments/assets/fb688bb1-92fa-4f5b-854d-b45f3cb8cfaf)

![image](https://github.com/user-attachments/assets/82673696-c9ad-4661-bd12-a9cd91234775)

![image](https://github.com/user-attachments/assets/75882712-d802-462b-a7f5-21923a9028a2)

![image](https://github.com/user-attachments/assets/085fc489-1f1c-4909-b9cc-630de85de0a5)

![image](https://github.com/user-attachments/assets/3e2192fd-fd37-4792-8030-c62b499bc8c6)


### 4.5 Confusion metrics
![image](https://github.com/user-attachments/assets/782f3a05-01e0-44a1-a90e-3ee145303eac)

![image](https://github.com/user-attachments/assets/7f6d07df-d3d9-44dc-bf3b-4ec33fb59f35)

![image](https://github.com/user-attachments/assets/edbf7583-5c64-4f84-a778-623e6697cc4a)

![image](https://github.com/user-attachments/assets/f43d216b-f076-4509-90d5-e737722cdcfd)


### 4.6 Performance metrics
Model	Accuracy (%)		Precision	Recall	F1-score

CNN	
96	No tumor	0.96	0.97	0.96
		Tumor	0.96	0.95	0.96

VGG16	
99	No tumor	0.98	1.00	0.99
		Tumor	1.00	0.97	0.99

VGG19	
99	No tumor	0.98	1.00	0.99
		Tumor	1.00	0.97	0.99

ResNet101	
89	No tumor	0.89	0.90	0.90
		Tumor	0.88	0.86	0.87

### 4.7 Prediction
![image](https://github.com/user-attachments/assets/454d746c-5a91-42c2-9199-982265bd6db5)

![image](https://github.com/user-attachments/assets/73e8283f-c5b2-4e27-b8a8-ae494ed16b9d)

![image](https://github.com/user-attachments/assets/33bfe193-0d85-4989-80ae-0337996b69fc)

![image](https://github.com/user-attachments/assets/987b478f-0306-447c-9f2a-67453f79dbb9)

## Chapter 5: Conclusion
The analysis revealed that the VGG19 model achieved the highest accuracy 98.74 among the tested architectures, demonstrating its strong potential for brain tumor prediction. However, all models exhibited signs of overfitting, as evidenced by discrepancies between training and validation/test performance metrics. This overfitting condition highlights the need for further refinement.
In future iterations, we plan to address overfitting by training the models on a larger and more diverse dataset. This approach aims to improve generalization and enhance prediction accuracy, making the models more robust for real-world applications. By incorporating these improvements, we anticipate producing a more reliable and effective solution for brain tumor detection.


###	Academic Project
