# tumor_detection


## Chapter 1: Introduction

### 1.1 About
A brain tumor is an abnormal growth of cells in the brain that can disrupt normal brain function. Tumors are classified as either benign (non-cancerous) or malignant (cancerous) and may originate within the brain itself, known as primary tumors, or spread from cancers elsewhere in the body, termed secondary or metastatic tumors. Primary brain tumors encompass several types, including gliomas, which arise from supportive glial cells, meningiomas that develop from the brain’s protective lining, and medulloblastomas, common in children and originating in the cerebellum. Secondary tumors are often metastases from cancers such as lung, breast, or melanoma. The symptoms of a brain tumor vary depending on its location, size, and growth rate, but common symptoms include persistent headaches, seizures, visual or auditory changes, cognitive or memory issues, nausea, and even behavioral changes.
The exact causes of brain tumors are not fully understood, though factors like genetics (for instance, conditions like neurofibromatosis) and high exposure to ionizing radiation may increase risk. Diagnosis typically involves imaging tests such as MRI or CT scans, and sometimes a biopsy to analyze tumor cells and determine the best treatment strategy. Treatment varies according to tumor type, location, and severity and may involve surgery to remove as much of the tumor as safely possible, followed by radiation therapy to kill remaining cells, chemotherapy, or newer methods like targeted therapy and immunotherapy that help the immune system attack cancer cells. Brain tumor prognosis depends on various factors, including tumor type and early detection, which improves the chances of better outcomes.
### 1.2 History  
The history of brain tumor detection dates back to ancient times, with the earliest evidence found in skulls showing signs of trepanation, a procedure where holes were drilled to relieve symptoms, likely due to intracranial pressure or seizures. However, true understanding and documentation of brain tumors didn’t develop until much later. During the Renaissance, scientists and anatomists like Andreas Vesalius made groundbreaking strides in studying the human brain, laying the foundation for neurology. In the 18th and 19th centuries, physicians began recognizing and differentiating various neurological conditions, including tumors, though diagnostic limitations meant that most tumors were only discovered post-mortem.
The introduction of X-ray technology in the late 19th and early 20th centuries marked a revolutionary advance in brain tumor detection, allowing doctors to visualize dense structures within the head for the first time. However, X-rays provided limited detail, so their usefulness for detecting soft tissue brain tumors was minimal. In the mid-20th century, pneumoencephalography, a painful and invasive procedure involving the injection of air into the cerebrospinal fluid, was introduced, offering somewhat better imaging of brain structures.
A major breakthrough occurred with the development of computed tomography (CT) in the 1970s. This non-invasive imaging technique allowed for clear cross-sectional images of the brain and was a major advancement in early tumor detection. Soon after, magnetic resonance imaging (MRI) became available in the 1980s, transforming brain tumor detection due to its superior soft tissue contrast and ability to reveal detailed images of brain structures. MRI remains the primary imaging technique used for brain tumors today, especially with advancements in functional MRI and magnetic resonance spectroscopy, which allow for precise mapping of tumor locations, metabolism, and effects on surrounding brain tissue.
The introduction of positron emission tomography (PET) and advanced molecular imaging has further refined brain tumor detection, enabling doctors to differentiate between tumor types and assess tumor growth or response to treatment at a cellular level. Today, research is ongoing in fields like artificial intelligence and machine learning, where algorithms are being developed to improve detection accuracy and even predict tumor types from imaging data. These advancements represent a long journey from ancient, crude methods to sophisticated, highly accurate technologies that have significantly improved the diagnosis, treatment, and prognosis for brain tumor patients.
 
### 1.3 Demerit of traditional process
Traditional methods of detecting brain tumors were foundational but had significant limitations that impacted diagnosis and treatment. Early procedures like pneumoencephalography were highly invasive and required injecting air into the cerebrospinal fluid, which caused considerable discomfort and carried risks of infection and other complications. Imaging quality was also a major issue, as techniques like X-rays offered low-resolution images that could only reveal dense structures, such as bones, while failing to provide the soft tissue details essential for accurately locating brain tumors. Consequently, tumors were often detected only at advanced stages, as severe symptoms needed to be present before any diagnostic tests were performed. This late detection limited treatment options and reduced patient prognosis.
Furthermore, the poor clarity of early imaging methods increased the likelihood of misdiagnosis, as doctors struggled to distinguish tumors from other neurological conditions. Additionally, traditional imaging techniques exposed patients to high radiation levels, which posed further health risks, especially when repeated scans were necessary for monitoring tumor progression or treatment response. These methods also lacked the ability to provide functional information about the tumor, such as its activity or effects on surrounding brain tissue, which are crucial for precise treatment planning. Without real-time imaging to accurately guide surgeons, it was difficult to determine tumor boundaries, often leading to incomplete tumor removal or unintended damage to healthy tissue. These limitations meant that traditional brain tumor detection methods often resulted in less accurate diagnoses, painful procedures, and fewer effective treatment options, which affected both the quality of life and survival outcomes for patients. Advances in imaging, like MRI and CT scans, have since addressed many of these challenges, providing safer, more accurate, and less invasive options for detecting and managing brain tumors.
### 1.4 AI overcome
Artificial intelligence (AI) has significantly improved brain tumor detection and management, addressing many of the limitations of traditional methods. First, AI-enhanced imaging, using algorithms in MRI and CT scans, allows for highly detailed, three-dimensional visualizations of brain structures. AI can analyze these images with exceptional precision, identifying even the smallest abnormalities in soft tissue, which traditional X-rays and early imaging techniques could not detect. This helps in detecting tumors at earlier stages, increasing treatment options and improving patient prognosis.
AI also aids in reducing diagnostic errors by distinguishing between tumor types and other neurological conditions. Through machine learning, AI algorithms are trained on vast datasets of brain images, learning to recognize patterns and subtleties in tumor characteristics that may be missed by the human eye. This improves the accuracy of diagnosis and helps clinicians in creating more tailored treatment plans based on the specific type and behavior of the tumor.
Another advantage is AI’s ability to track tumor progression over time with precision. Traditional methods often required repeated imaging, which exposed patients to additional radiation. In contrast, AI algorithms can predict tumor growth patterns, assessing treatment response and minimizing the need for frequent scans. When imaging is necessary, AI algorithms help optimize imaging parameters to reduce radiation exposure without compromising image quality, thus enhancing patient safety.
AI-powered tools have also transformed surgical planning and guidance. By integrating real-time imaging data, AI can create a detailed map of the tumor and its impact on nearby brain structures. Surgeons can use this data to plan the safest, most effective surgical route, minimizing damage to healthy tissue and reducing the risk of incomplete tumor removal. Some AI systems even provide real-time assistance during surgery, identifying tumor boundaries and alerting surgeons to critical areas, making procedures less invasive and more accurate.
Finally, AI offers insight into the molecular and genetic makeup of brain tumors. Advanced AI techniques like deep learning can analyze imaging alongside molecular data, providing a comprehensive profile of the tumor. This can guide the choice of treatments, such as targeted therapies, personalized for the tumor’s specific characteristics. By addressing issues like late detection, diagnostic inaccuracy, radiation exposure, and surgical risk, AI has transformed brain tumor detection and management, making it safer, more precise, and more effective.
### 1.5 ML used to predict, and methods used
Machine learning (ML) plays a vital role in predicting brain tumors by analysing complex data from medical imaging and patient records. With ML, computers learn patterns and features in brain images that indicate tumors, assisting with detection and prediction of tumor type, size, growth rate, and potential treatment responses. ML models are particularly powerful for analysing MRI, CT, and other imaging data, detecting even subtle abnormalities in brain tissue that might suggest a tumor. Some ML models also integrate clinical and genetic data, offering a comprehensive assessment that helps with early detection and personalized treatment planning.
Among the various ML methods used for brain tumor prediction, Convolutional Neural Networks (CNNs) are particularly effective for image-based tasks. CNNs excel at recognizing patterns in images, identifying tumor regions, and distinguishing them from healthy tissue by learning from thousands of labelled scans. Popular CNN architectures like ResNet and U-Net are commonly used for precise tumor segmentation, allowing clinicians to clearly outline the tumor’s boundaries. Support Vector Machines (SVMs) are another ML technique widely used in brain tumor classification, particularly for distinguishing between benign and malignant tumors. By finding optimal boundaries that separate data points in high-dimensional space, SVMs can classify tumors based on specific imaging features, aiding in accurate diagnosis.
Random Forests, an ensemble learning method that combines multiple decision trees, are used for classifying tumor types and can integrate data beyond imaging, such as patient demographics and clinical records, to improve prediction accuracy. The K-Nearest Neighbors (K-NN) algorithm, while simpler, also sees some use in tumor classification. It compares a tumor characteristic with its "neighbors" in a labelled dataset, assigning the most common label among the closest matches. Artificial Neural Networks (ANNs) are also widely used due to their ability to learn complex relationships between features in imaging and non-imaging data, making them versatile for classifying tumors and predicting outcomes like tumor size or growth rate. Finally, advanced deep learning architectures, including Recurrent Neural Networks (RNNs) and Autoencoders, are being applied to analyze sequential data or reduce complex imaging data to its most essential features, enhancing predictive accuracy further.
Overall, ML methods in brain tumor prediction provide a level of precision that traditional methods lacked, facilitating early detection and more effective treatment planning. With ongoing advancements, these ML techniques continue to improve, enhancing the reliability and accuracy of brain tumor predictions in clinical settings.
### 1.6 Demerit of ML
Machine learning (ML) in brain tumor detection, while promising, has several limitations and challenges. One of the primary issues is the dependency on large, high-quality datasets to train effective models. Brain tumor detection requires extensive labelled medical imaging data, which can be difficult to collect due to 
privacy concerns, high imaging costs, and the relative rarity of certain tumor types. Without such data, ML models may struggle to generalize, which can lead to inaccuracies when detecting or classifying tumors across diverse patient populations. This reliance on data quality and availability is a significant barrier to the accuracy and reliability of ML in clinical settings.
Additionally, the complexity of many ML models, especially deep neural networks, presents an interpretability challenge. These models often function as "black boxes," making it difficult to understand how they arrive at specific predictions. In medical applications, this lack of transparency can be problematic, as doctors may be hesitant to rely on a model’s predictions without knowing the underlying rationale. Errors in ML predictions can lead to misdiagnosis or improper classification of tumors, potentially resulting in ineffective or harmful treatments for patients, which makes interpretability an essential consideration.
ML models are also vulnerable to biases present in their training data. If a dataset overrepresents certain demographics or imaging techniques, the model may struggle to accurately predict tumor characteristics in underrepresented populations, thereby limiting its clinical utility across diverse groups. This lack of inclusivity can reduce the model’s effectiveness in real-world settings. Furthermore, ML models require frequent updates to adapt to new data or advancements in imaging technology. Without consistent retraining, ML models can become outdated, which risks decreasing their prediction accuracy and reliability over time.
Finally, implementing ML in clinical environments poses technical and logistical challenges. ML systems often require specialized hardware, integration with medical imaging technology, and compliance with healthcare regulations to ensure patient privacy and safety. These factors can make it challenging for hospitals and clinics, especially those with limited resources, to adopt ML-based tumor detection solutions. In summary, while ML offers valuable advancements in brain tumor detection, challenges such as data dependence, model interpretability, potential bias, and technical barriers must be addressed to fully leverage its benefits in clinical practice.
### 1.7 Deep Learning is used to overcome
Deep learning, a subset of machine learning, addresses many of the limitations of traditional ML methods in brain tumor detection by utilizing complex neural networks that are particularly well-suited for analysing large, intricate datasets, such as medical images. Unlike conventional ML models, which often rely heavily on handcrafted features or require extensive pre-processing, deep learning algorithms, especially convolutional neural networks (CNNs), can automatically learn relevant features directly from raw data, such as MRI or CT scans. This capability allows deep learning models to capture finer details within images, leading to more accurate and reliable tumor detection, even in cases where traditional ML might struggle due to limited or variable quality data.
One of the major advantages of deep learning is its ability to handle vast amounts of data, which addresses the problem of data dependency that traditional ML models face. By training on large and diverse datasets, deep learning models can achieve a higher level of generalization, improving their performance across varied populations. Moreover, the deep learning approach can integrate multiple data types, such as imaging data and clinical records, which offers a more comprehensive analysis of each patient’s unique condition and further enhances the accuracy of predictions and classifications.
Deep learning also tackles the interpretability issue in traditional ML through advanced techniques like heatmaps and saliency maps, which highlight the areas of an image that the model focuses on while making its predictions. These visualization tools help clinicians understand the reasoning behind a model’s decision, offering more transparency and confidence in its findings. Although deep learning models remain complex, these interpretability aids make them more user-friendly in clinical environments and help ensure that clinicians can trust and validate the predictions, thereby reducing the risk of misdiagnosis.
Another benefit of deep learning is its resistance to data biases due to its ability to learn robust patterns from diverse datasets. Techniques like data augmentation, where the model is trained on rotated, flipped, or cropped versions of images, help improve its generalization and prevent overfitting to specific demographics or imaging conditions. This flexibility makes deep learning models more effective across a broader range of cases, thus increasing their clinical utility.
### 1.8 The Models used
Method Name |	Accuracy (%)
CNN |	96.01
VGG-16 |	98.71
VGG-19 |	98.74
ResNet-101 |	88.58

We compared three models CNN, ResNet-101, and VGG-16 evaluating their performance and limitations for brain tumor detection. The baseline CNN, though effective, has a simpler architecture with fewer layers, making it less capable of capturing the intricate patterns often necessary for precise tumor identification in complex, high-dimensional medical images. This lack of depth limits its ability to distinguish subtle differences, reducing accuracy when handling diverse tumor types or stages.
VGG-16, a deep convolutional neural network, provides a balance between simplicity and depth. With its 16 layers, it has shown strong feature extraction capabilities, making it effective in identifying patterns within medical images. However, compared to VGG-19, VGG-16 has slightly fewer layers, which limits its ability to capture finer details in brain tumor images, potentially reducing its effectiveness in identifying smaller or less distinct tumor regions.
ResNet-101, on the other hand, is a very deep network that incorporates residual connections to address the vanishing gradient problem, allowing it to learn efficiently even with many layers. While ResNet-101 is powerful, its complex structure requires significant computational resources and can sometimes lead to challenges with interpretability. Additionally, its high number of layers and residual blocks increase the risk of overfitting when trained on limited datasets, which is a common constraint in medical imaging.
In light of these observations, we propose VGG-19 as an improved model for brain tumor detection. With its additional layers, VGG-19 offers enhanced feature extraction while maintaining a straightforward, interpretable architecture similar to VGG-16. This deeper structure allows VGG-19 to capture more intricate patterns, improving its ability to detect subtle tumor features and thus enhancing overall prediction accuracy. By addressing the specific gaps identified in CNN, ResNet-101, and VGG-16, VGG-19 is a more robust choice, better suited for accurate, generalizable, and clinically valuable tumor detection.








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
