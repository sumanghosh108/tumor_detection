### tumor_detection


## Chapter 4: Proposed work
#4.1 Experimental setup
#4.1.1 Computer configuration
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
(Fig. 10: Flow chart of execution process)
#4.4 Output
Model		Training (%)	Validation (%)	Test (%)

CNN	Accuracy	99.33	86.71	96.01
	Loss	2.10	77.60	18.76

VGG-16	Accuracy	100	95.71	98.71
	Loss	0.12	15.33	5.39

VGG-19	Accuracy	100	95.79	98.74
	Loss	0.25	15.43	4.72

ResNet-101	Accuracy	90.84	80.28	88.58
	Loss	23.22	46.13	28.55
(Table 2: Comparison between proposed models with respect to train, test and validation)

![image](https://github.com/user-attachments/assets/1cc5adbc-7abd-4cb4-8042-953fee5757fb)
(Fig. 11: Comparison of trained models accuracy graph with respect to train and validation)
![image](https://github.com/user-attachments/assets/d5fd3ae2-792e-4a80-9b2c-cc587698b819)
(Fig. 12: Comparison of trained models loss graph with respect to train and validation)
![image](https://github.com/user-attachments/assets/1a70b074-8f0d-4638-a998-b2d3c291ba1c)
(Fig. 13: CNN accuracy graph)
![image](https://github.com/user-attachments/assets/18385168-a01e-458a-8aa5-6e58da641e2d)
(Fig. 14: CNN loss graph)
![image](https://github.com/user-attachments/assets/ff074375-ceac-4223-b314-0606cdadf4cc)
(Fig. 15: VGG-16 accuracy graph)
![image](https://github.com/user-attachments/assets/fb688bb1-92fa-4f5b-854d-b45f3cb8cfaf)
(Fig. 16: VGG16 loss graph)
![image](https://github.com/user-attachments/assets/82673696-c9ad-4661-bd12-a9cd91234775)
(Fig. 17: VGG19 accuracy graph)
![image](https://github.com/user-attachments/assets/75882712-d802-462b-a7f5-21923a9028a2)
(Fig. 18: VGG19 loss graph)
![image](https://github.com/user-attachments/assets/085fc489-1f1c-4909-b9cc-630de85de0a5)
(Fig. 19: ResNet101 accuracy graph)
![image](https://github.com/user-attachments/assets/3e2192fd-fd37-4792-8030-c62b499bc8c6)
(Fig. 20: ResNet101 loss graph)

# 4.5 Confusion metrics
![image](https://github.com/user-attachments/assets/782f3a05-01e0-44a1-a90e-3ee145303eac)
(Fig. 21: Confusion matrix of CNN)
![image](https://github.com/user-attachments/assets/7f6d07df-d3d9-44dc-bf3b-4ec33fb59f35)
(Fig.22: Confusion matrix of VGG16)
![image](https://github.com/user-attachments/assets/edbf7583-5c64-4f84-a778-623e6697cc4a)
(Fig.23: Confusion matrix of VGG19)
![image](https://github.com/user-attachments/assets/f43d216b-f076-4509-90d5-e737722cdcfd)
(Fig. 24: Confusion matrix of ResNet101)

#4.6 Performance metrics
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

#4.7 Prediction
![image](https://github.com/user-attachments/assets/454d746c-5a91-42c2-9199-982265bd6db5)
(Fig. 25: Prediction using CNN)
![image](https://github.com/user-attachments/assets/73e8283f-c5b2-4e27-b8a8-ae494ed16b9d)
(Fig. 26: Prediction using VGG16)
![image](https://github.com/user-attachments/assets/33bfe193-0d85-4989-80ae-0337996b69fc)
(Fig. 27: Prediction using VGG19)
![image](https://github.com/user-attachments/assets/987b478f-0306-447c-9f2a-67453f79dbb9)
(Fig. 28: Prediction using ResNet101)



