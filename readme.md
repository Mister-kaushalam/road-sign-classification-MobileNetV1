<H1>Road Sign Classification using MobileNetV1 </H1>

- Author - Kaushal Bhavsar

Model weights can be downloaded from here: [Link](https://drive.google.com/file/d/1T4Rlba1FVLHiGtIRTeYAhgykLZYsOjf9/view?usp=sharing)

**Table of Contents**

1. [Data Set](#_page2_x72.00_y63.00)
   
   1.1 [Visualization](#_page2_x72.00_y589.42)
   
   1.2 [Feature Selection / Engineering](#_page5_x72.00_y522.63)
   
   1.3 [Correlation](#_page6_x72.00_y189.95)

2. [Data Pre-Processing](#_page6_x72.00_y376.60)

3. [Network Structure and Other Hyperparameters](#_page7_x72.00_y309.88)
   
   3.1 [Walkthrough of MobileNet Model’s Code](#_page9_x72.00_y625.37)
   
   3.2 [Details of the Created Model](#_page11_x72.00_y95.30)
   
   3.3 [Training of MobileNet Model](#_page11_x72.00_y234.35)
   
   3.4 [Model Testing](#_page12_x72.00_y335.50)

4. [Cost / Loss / Error / Objective Function](#_page12_x72.00_y569.76)

5. [Optimizer](#_page14_x72.00_y362.36)

6. [Cross Fold Validation 13](#_page14_x72.00_y543.66)

7. [Results 14](#_page15_x72.00_y417.84)

   7.1 [Accuracy 14](#_page15_x72.00_y467.64)

   7.2 [Mirco Precision, Micro Recall, and Micro F1 Score 15](#_page16_x72.00_y631.98)

8. [Evaluation of Results 16 ](#_page17_x72.00_y285.40)

9. [Impact of Various Parameters / Hyperparameters 17](#_page18_x72.00_y269.29)

   9.1 [Hyper-Parameter Tuning Code walkthrough 17](#_page19_x72.00_y63.00)
   
   9.2 [Overfitting 18](#_page19_x72.00_y469.68)

   9.3 [Underfitting 19](#_page21_x72.00_y63.00)

10. [References 21](#_page22_x72.00_y123.83)

# <a name="_page2_x72.00_y63.00"></a>**1. Data Set**

The Convolutional Neural Network (CNN) is a type of neural network primarily used to take in, process, and classify data in an image format. As a result, it was prudent to find an appropriate dataset of images for classification. Kaggle, an open-source data repository, was used to find a multi-class dataset.

For this project, the German Traffic Sign Recognition Benchmark (GTSRB) was used as the primary dataset. The dataset has over fifty thousand single images of various road signs, where each image can be classified into more than forty different classes. An example of the type of included images in this multi-class dataset is shown in Figure 1.

<figure>
<img src="report images/sign-1.png" alt="Image description">
  <figcaption>Figure 1: Example traffic sign image from the GTSRB dataset.</figcaption>
</figure>


Specifics of the datasets are as below:

1. Single-image, multi-class classification problem
1. A total of 43 different classes
1. Training data contains 39,209 images and uses a 20% split:
   1. 31367 images are taken for training
   1. 7842 images are taken for validation
1. Testing data contains 12630 images in total.

We have created a CustomDataset(Dataset) class which inherits the Dataset class from pytorch.utils.data. The CustomDataset class is expected to provide easy access to images along with their bounding boxes and class labels.

The specifics of the dataset, including visualization of attributes, feature selection, and the implication of correlation, are discussed in detail below.

## 1.1 *Visualization*

<a name="_page2_x72.00_y589.42"></a>An initial experimental analysis and visualization of the data was an important starting point to verify that this was a usable and fitting dataset.

The Kaggle data set was split preemptively into test and train sections, with the images divided between two folders. Accompanying the split data are two separate CSV files for the train and test images. The CSV files list all individual images, metadata, and additional information.

Table 1: Five Entries in the Original Train Data CSV File.
<figure>
<img src="report images/train-data-csv.png" alt="Image description">
  <figcaption>The *Width* and *Height* columns in Table 1 represent the respective width and height of the image. Whereas, for the bounding box or region of interest (ROI), the coordinate position at the top left corner is defined by the variables *Roi.X1* and *Roi.Y1*, and the coordinate position at the bottom right corner is represented by the variables *Roi.X2* and *Roi.Y2*. The *ClassId* column holds an integer value representing the traffic sign class that the image belongs to. Lastly, the column *Path* holds the relative file path of the image.</figcaption>
</figure>



Table 2: Five Entries in the Test Data CSV File.
<figure>
<img src="report images/test-data-csv.png" alt="Image description">
  <figcaption>The Test Data CSV file, as seen in Table 2, was structured similarly to the Original Train Data CSV file shown in Table 1. The columns are the same for both files and represent the same thing value-wise.</figcaption>
</figure>

<figure>
<img src="report images/test-data-csv.png" alt="Image description">
  <figcaption>Figure 2: Different traffic sign image classes found in the GTSRB dataset [1].</figcaption>
</figure>

After the data was separated into training, test, and validation sets, the total amount of images set aside for training was 31,367. For validation, there were 7,842 images available, and the total amount of images for testing was 12,630.

The distribution of the forty-two different traffic sign classes for the three data sets: training set, validation set, and test set are shown in Figure 3, Figure 4, and Figure 5, respectively.

<figure>
<img src="report images/train-set-class-dis.jpeg" alt="Image description">
  <figcaption>Figure 3: Traffic Sign Class Distribution for the Training Set.</figcaption>
</figure>

<figure>
<img src="report images/val-set-class-dis.jpeg" alt="Image description">
  <figcaption>Figure 4: Traffic Sign Class Distribution for the Validation Set.</figcaption>
</figure>

<figure>
<img src="report images/test-set-class-dis.jpeg" alt="Image description">
  <figcaption>Figure 5: Traffic Sign Class Distribution for the Test Set.</figcaption>
</figure>


Overall, the distribution of traffic sign classes across the training, validation, and test data sets is non-linear. There is indeed some class imbalance but should be fine for our use case.

## 2. *Feature<a name="_page5_x72.00_y522.63"></a> Selection / Engineering*

Feature selection is an extremely important technique in a general sense; it can also play an essential role in the data pre-processing approach. Providing relevant and useful data to the model is critical as it could reduce the chance of overfitting. As a result, it's an essential method to use when a dataset has a lot of irrelevant features that would unnecessarily create complexity in the model.

This project did not utilize a specific feature selection technique with the GTSRB dataset. The decision occurred after analyzing the data and realizing that the features of the data were important for either object detection (i.e., bounding box, image location, image dimensions) or object classification and would cause more issues if removed.

Besides feature selection, feature engineering is another extremely important technique. Unlike feature selection, which potentially removes features, feature engineering allows for the creation of new variables that don't exist in the data set.

Similarly to before, the feature engineering technique was not utilized with the GTSRB dataset. The decision occurred after analyzing the data and coming to an understanding that the pre-existing features of the data were more than enough for classification. In addition, the decision to avoid an overly complex dataset was also a factor.

## 3. *Correlation*

<a name="_page6_x72.00_y189.95"></a>Correlation, in a general sense, compares the relationship between two variables. Recognizing correlation in a data set is extremely important. It's well known that data features that are highly correlated result in an observable shared linear relationship.

Finding correlations in image datasets is challenging due to the complex, nonlinear relationships between pixels, variability in appearance, and difficulties in extracting meaningful features, making traditional correlation analysis less effective. Hence, we decided to not perform feature correlation as we relied on the Neural Network to decide the feature importance by updating the relevant weights during training.

# 2. **Data<a name="_page6_x72.00_y376.60"></a> Pre-Processing**

Data pre-processing is a critical step for preparing data for use with models and involves converting the raw data into a usable format.

We created augmentation pipelines using the Albumentations library for the training, validation, and testing datasets. For the training dataset (train\_augs), a single transformation is applied, namely resizing images to dimensions of 224x224 pixels. This augmentation is crucial for standardizing the input sizes across the dataset, ensuring consistency during the training process. Moreover, the bounding box parameters are defined per the Pascal VOC format, incorporating class IDs within the label fields. An example of a traffic sign image with a bounding box is shown in Figure 6.

<figure>
<img src="report images/stop-sign.jpeg" alt="Image description">
  <figcaption>Figure 6: Example of a traffic sign image with a bounding box around the sign.</figcaption>
</figure>

Similar to the training augmentation pipeline, the validation (val\_augs) and testing (test\_augs) pipelines are also configured to resize images to the standardized dimensions of 224x224 pixels.

# 3. **Network<a name="_page7_x72.00_y309.88"></a> Structure and Other Hyperparameters**

MobileNets was the convolutional neural network chosen and implemented for the project. Introduced in 2017 by Google and employed by researcher Andrew G. Howard, MobileNets are well-designed models that can be used in a variety of different settings and use cases, including object detection and classification [3]. The original open-source model focused on providing an option to the computer vision community that was both accurate in its methods and not too expensive resource-wise to use. The Google researchers achieved this through the use of two hyper-parameters and an efficient network architecture design [4].

Though each MobileNet generation has its benefits and drawbacks, the specific model used for the project was the first MobileNet version released in 2017. The MobileNet architecture implemented was based on an open-source Kaggle notebook [5], and a summary of the model and its structure are discussed below.

Table 3: Body Architecture for MobileNet Model [4].

<figure>
<img src="report images/model-layers.jpeg" alt="Image description">
  <figcaption>Table 3 depicts the MobileNet model architecture used in the project. As observed in the table, the model has a total of twenty-eight layers and each layer has its own specific filter shape and input size. Of course, pointwise and depthwise convolutions count as different layers, and the fully connected layer, which, together with the rest of the convolution layers, is where the twenty-eight value comes from [5]. The convolution layers themselves make up a large portion of the layers with several 3x3 depthwise convolution and 1x1 convolution layers occurrences, as well as one instance of a 3x3 convolution.</figcaption>
</figure>


Overall, the MobileNet architecture is straightforward, and the layers match up with what a typical convolutional neural network (CNN) is composed of: features such as an input, output, convolution layers, pooling layer(s), and fully connected layer.

Table 4: Distribution of Resources Per Layer Type [4].

<figure>
<img src="report images/dw-layer.png" alt="Image description">
</figure>

As for the number of parameters and multiply-adds (i.e., Mult-Adds) per layer, Table 4 shows the values for each of the four most prominent layer types. The 1x1 convolution layers have both the highest percentage of parameters and multiply-adds.

As mentioned previously, the MobileNet architecture involves several 3x3 depthwise convolution layers. The main theory and reason why this specific type of convolution is utilized is because it lowers the number of parameters used (e.g., see results in Table 3). In addition, they also reduce the overall number of computations used in convolutional operations [6].

Table 5: MobileNet Comparison to Popular Models [4].

<figure>
<img src="report images/model-description.png" alt="Image description">
</figure>

The structural compactness of MobileNet makes it suitable for a road sign classification use case as these models will be implemented on mobile devices like vehicles with limited computation resources. Hence, we selected this model.

## 1. *Walkthrough<a name="_page9_x72.00_y625.37"></a> of MobileNet Model’s Code:*

The code in “Create the MobileNet Model Class” defines a custom MobileNet model architecture using PyTorch's neural network module[9]. Here's a concise explanation: Three different types of convolutional blocks are defined: Convolution\_Block, Depth\_Convolution\_Block, and Convolution\_Block\_Same. Each block consists of convolutional layers followed by batch normalization and ReLU activation.

The MyMobileNet class defines the MobileNet architecture. It comprises two sets of convolutional layers (layers1 and layers2). The first set contains a series of convolutional and depthwise separable convolutional blocks, followed by batch normalization and ReLU activation.

To streamline the creation of multiple layers, lists (module\_list\_depth and module\_list\_conv) are used to hold instances of the depthwise and regular convolutional blocks, respectively. These lists enable easy manipulation of the number of layers for architectural experimentation, facilitating adjustments to engineer overfitting or underfitting.

The second set of layers includes additional depthwise and regular convolutional blocks, followed by adaptive average pooling, flattening, and a fully connected layer to produce the final output.

In the forward method, the input data passes through the first set of layers (layers1), followed by the depthwise and regular convolutional blocks defined in the lists. Finally, the data passes through the second set of layers (layers2) to generate the output predictions.

The softmax function is **not** explicitly used in the model's layers because the Cross-Entropy Loss function (nn.CrossEntropyLoss()) incorporates the softmax operation internally.

In PyTorch, when using nn.CrossEntropyLoss(), the softmax operation is applied to the raw output logits of the last layer during the calculation of the loss. This means that the model architecture itself does not need to include a softmax layer explicitly.

The overall design of the MobileNet model aims to strike a balance between computational efficiency and performance, making it well-suited for deployment on resource-constrained devices or applications requiring real-time inference.

```
model = MyMobileNet(3, classes = 43)
model.to(device)
summary(model, (3,224,224))
```

This code initializes an instance of the MyMobileNet model with input channels set to 3 (RGB) and output classes set to 43. It then moves the model to the specified device (e.g., CPU or GPU) using the to() method. Afterward, the summary() function from the torch summary library is called to generate a summary of the model architecture.

## 2. *Details<a name="_page11_x72.00_y95.30"></a> of the Created Model*

1. Total number of layers: 29
1. Total Trainable parameters: 30,492,203
1. Input size (MB): 0.57
1. Forward/backward pass size (MB): 107.06
1. Params size (MB): 116.32
1. Estimated Total Size (MB): 223.95

## 3. *Training<a name="_page11_x72.00_y234.35"></a> of MobileNet Model*

The training and validation loops for the model are executed over multiple epochs. Before commencing the loops, lists are initialized to store training and validation metrics, including losses and accuracies. Additionally, variables are set to track the best validation loss observed so far (best\_val\_loss) and to store the weights of the best-performing model (best\_model\_weights), initialized as None.

The training loop (for epoch in range(1, epochs + 1)) iterates over each epoch, with the model set to training mode (model.train()). Within this loop, the training data is processed in batches using a progress bar (tqdm(train\_loader)). For each batch, the input images and labels are transferred to the appropriate device (e.g., GPU), and a forward pass through the model is performed to obtain classification outputs. The classification loss is computed using the specified loss function (criterion) and backpropagation is applied to update the model parameters using the Adam optimizer (optimizer). Training metrics such as loss and accuracy are accumulated and updated.

Similarly, the validation loop evaluates the model's performance on the validation dataset. The model is set to evaluation mode (model.eval()), and the validation data is processed batch-wise using a progress bar. Gradient computation is disabled (torch.no\_grad()) to speed up inference. The classification loss is computed for each batch, and validation metrics are accumulated. We have noted the total training time in the results section.

<figure>
<img src="report images/loss-acc-epochs.jpeg" alt="Image description">
  <figcaption>Figure 7: Accumulated loss and accuracy over 15 epochs of training and validation cycle.</figcaption>
</figure>


## 3.4. *Model<a name="_page12_x72.00_y335.50"></a> Testing*

A function named "test\_model()" is defined to evaluate the model's accuracy on the test data. This function takes the trained model and the test\_loader as inputs. During evaluation, the function iterates through the test data batches, computes predictions using the model, and compares them with the ground truth labels. It calculates accuracy, precision, and recall scores using scikit-learn's metrics functions. Additionally, it computes the precision-recall curve, its area under the curve (AUC-PR), and returns these metrics.

After defining the function, it is called to evaluate the model on the test data. The obtained test accuracy is printed, and a precision-recall curve is plotted using Matplotlib, displaying the relationship between precision and recall values. The AUC-PR score is also annotated on the plot for reference. This comprehensive evaluation provides insights into the model's performance, particularly in scenarios where class imbalances exist, as indicated by precision-recall metrics.

# 4. **Cost<a name="_page12_x72.00_y569.76"></a> / Loss / Error / Objective Function**

The loss function observes the difference between the predictions and the actual values. The loss function, referred to as the objective, error, or cost function, are four terms used interchangeably with each other [8].

The loss function is specified as the Cross-Entropy Loss (nn.CrossEntropyLoss()), which is commonly used for multi-class classification tasks. This loss function computes the softmax activation internally, making it suitable for scenarios where the output is class probabilities.

Several other loss functions were researched and considered during the process, including Mean Square Error (MSE) and Mean Absolute Error (MAE). Though these functions are beneficial in their own right, they are not the most optimal choice when working with image classification, so the Cross Entropy Loss function was best suited for the given scenario.

<figure>
<img src="report images/loss-over-epocs.jpeg" alt="Image description">
  <figcaption>Figure 8: Training and Validation Loss Plot using Cross Entropy.</figcaption>
</figure>

<figure>
<img src="report images/loss-over-epocs-zoom.jpeg" alt="Image description">
  <figcaption>Figure 9: Training and Validation Loss Plot using Cross Entropy.</figcaption>
</figure>

# 5. **Optimiser**

<a name="_page14_x72.00_y362.36"></a>Optimizers are an essential component of neural networks and their performance. As such, there are a variety of optimizers that exist, each with its benefits and drawbacks.

For optimization, the Adam optimizer (optim.Adam) is employed. Adam is an adaptive learning rate optimization algorithm that combines the advantages of both AdaGrad and RMSProp. It adapts the learning rate for each parameter individually, providing faster convergence and better generalization. The learning rate is set to 0.001, which is selected arbitrarily but we’ll employ hyper-parameter tuning to find the optimal learning rate later in the code.

# 6. **Cross<a name="_page14_x72.00_y543.66"></a> Fold Validation**

Cross-validation is an important technique utilized as a preventative measure to avoid overfitting and can evaluate a model's performance. Several different methods exist, including, but not limited to, k-fold, stratified cross-validation, leave-one-out cross-validation, and Holdout validation [7].

Though these alternative methods were considered, the K-fold cross-validation was chosen as it was the cross-validation technique best suited for the given problem.

In general, K-fold cross-validation is used to evaluate the performance of the model across different folds of the dataset.

<figure>
<img src="report images/k-fold-val-acc.png" alt="Image description">
  <figcaption>Figure 10: Validation Accuracy Plots for K(10)-folds.</figcaption>
</figure>


The validation accuracy for each fold is stored in the fold\_val\_accuracies list, and the standard deviation of these accuracies across all folds is computed using NumPy's np.std() function.

Variations in the accuracies during the 10-fold cross-validation = 0.0043. This is an extremely low variation in the accuracy. This signifies that the model is able to generalize well.

# 7. **Results**
## 1. <a name="_page15_x72.00_y417.84"></a>*Accuracy*

<a name="_page15_x72.00_y467.64"></a>Accuracy was a metric that was recorded and monitored during the entirety of the project as it is a good indicator of how the model is performing and shows its ability to classify the traffic sign images correctly.

Accuracy was measured during three main stages: while the model was trained, during validation, and testing. The observed results are shown below.

Figure 11 below shows the observed training and validation accuracy values over a specified period of time (epochs). The range of resulting accuracy scores is satisfactory and the fact that the training and validation graphs are very close to one another in value is promising.

<figure>
<img src="report images/acc-over-epochs.jpeg" alt="Image description">
  <figcaption>Figure 11: Training and Validation Accuracy Plots.</figcaption>
</figure>

Figure 12 shows the observed training and testing accuracy values over a specified period of time (epochs). The recorded scores took place after the model was trained and tested on the test data set. The test accuracy remained constant at a value of 0.94972288202692, while the training graph approached an accuracy score of 1.

<figure>
<img src="report images/train-test-acc.jpeg" alt="Image description">
  <figcaption>Figure 12: Training and Validation Accuracy Plots.</figcaption>
</figure>


## 2. *Mirco<a name="_page16_x72.00_y631.98"></a> Precision, Micro Recall, and Micro F1 Score*

Since we have a multi-class problem, we cannot just calculate precision and recall. Additionally, due to class imbalance in our data, we have to consider using micro precision, micro recall, and micro F1 to gauge our model’s performance. Micro precision measures the proportion of correctly predicted instances across all classes, considering false positives and true positives.

Similarly, micro recall evaluates the model's ability to capture all relevant instances across all classes, accounting for false negatives and true positives. Micro F1 score, the harmonic mean of micro precision and micro recall, offers a balanced assessment of the model's performance, especially in scenarios where class imbalance is prevalent.

For our MobileNet model, we get

- Micro Precision: 0.9495645288994458
- Micro Recall: 0.9495645288994458
- Micro F1 Score: 0.9495645288994458
- Test Accuracy: 0.9495645288994458

# <a name="_page17_x72.00_y285.40"></a>**8 Evaluation of Results**

Overall, the MobileNet performed well on the German Traffic Sign Recognition Benchmark (GTSRB) dataset. A high percentage of the images during the different phases were classified correctly. Additional parameters were in play, and these elements contributed to the observed success of the model.

Based on both Figure 11 and Figure 12**,** the accuracy of the model during the training, testing, and validation stages increased steadily over increasing epochs and showed that the MobileNet model performed well, accuracy-wise, for image classification. In addition, a test accuracy score of 0.94 shows that the model generalizes new data well and can recognize patterns sufficiently.

F1 score values of approximately 0.95 indicate that the model performs well across all classes, achieving a high level of precision, recall, and overall F1 score. This suggests that the model effectively identifies instances from each class, maintaining a balance between precision and recall, resulting in high accuracy.

A micro precision of approximately 0.95 indicates that around 95% of the instances predicted as positive by the model were indeed correct across all classes. With a micro recall value of approximately 0.95, it implies that around 95% of all actual positive instances across all classes were successfully identified by the model. In other words, the model has a high level of precision and recall in its predictions, effectively minimizing the number of false positives and false negatives.

Variations in the accuracies during the 10-fold cross-validation is approximately 0.0043. This is an extremely low variation in the accuracy. This signifies that the model can generalize well to unseen data.

We undertook a comparative analysis of the runtime performance of MobileNet across Intel i9 CPU and Nvidia RTX 4080 GPU hardware, executing the model for a total of ten epochs. Initially executed on the Intel i9 CPU, the model incurred a training execution runtime of approximately twenty-five minutes per epoch, and validation time took about five minutes per epoch. In total, model execution on the CPU took approximately **seven hours**. Upon migration of the execution to the Nvidia RTX 4080 GPU, a significant enhancement in runtime efficiency was observed, with the model completing ten epochs in approximately **twenty minutes** in total. On the GPU, training, and validation epochs were completed in approximately 1.5 minutes and thirty seconds per epoch, respectively.

# 9. **Impact<a name="_page18_x72.00_y269.29"></a> of Various Parameters / Hyperparameters**

Hyperparameters are highly impactful on the behavior of a neural network, including the model's performance (i.e., accuracy, time taken, etc.). As a result, the hyperparameters must be optimally set. Suboptimal or inadequately chosen hyperparameters could cause further issues, such as poor generalization or ill-fitting models.

Table 6: Best Hyperparameters After Tuning.



|**Hyperparameters**|**Value**|
| - | - |
|Best Accuracy Achieved|0\.9918388166284111|
|Best Learning Rate|0\.00015457213645672622|
|Best Regularization Achieved|0\.0005272508590782009|

After careful hyperparameter finetuning and testing, Table 6 shows the best accuracy, learning rate, and regularization values observed.

These values were achieved after finding a balance between parameters such as model complexity, the size of the dataset, and other factors. Though the model is acting in the best-observed state now, if these few parameters were to change significantly, the model may fail and start overfitting or underfitting the data.

## 1. *Hyper-Parameter<a name="_page19_x72.00_y63.00"></a> Tuning Code Walkthrough*

The hyper-parameter tuning code defines a function train\_model() that trains a model with specified hyperparameters (lr, weight\_decay, optimizer\_type) and returns the accuracy achieved on the validation dataset. Within the function, the optimizer is dynamically chosen based on the optimizer\_type parameter, where 'Adam' corresponds to the Adam optimizer and 'SGD' corresponds to Stochastic Gradient Descent (SGD). The loss function is set to Cross Entropy Loss.

The training loop runs for a fixed number of epochs (5 in this case), where the model is set to training mode, and the optimizer is used to optimize the model parameters based on the computed loss.

After training, the model's performance is evaluated on the validation dataset, and the accuracy is computed. This accuracy value serves as the metric to be optimized by the **Bayesian optimization** process.

Next, the bounds for hyperparameters (pbounds) are defined, specifying the range of values for lr and weight\_decay. Bayesian optimization is initialized using the BayesianOptimization class from the bayes\_opt library, with the function to optimize (f=train\_model), hyperparameter bounds, and a random seed for reproducibility.

The optimization process is then executed using the maximize() method of the optimizer object. It starts with a specified number of initial random points (init\_points=5) to explore the hyperparameter space, followed by a specified number of iterations (n\_iter=5) to refine the search and find the optimal hyperparameters.

## 2. *Overfitting*

<a name="_page19_x72.00_y469.68"></a>The overarching goal was to see how these parameters affected the MobileNet model. The first step was to engineer overfitting purposefully. After careful research and deliberate testing, four main model elements were investigated and changed.

To start, the model's complexity was significantly increased and involved **increasing the number of depthwise convolutions and convolution layers**. The reason why this had a hand in causing the model to overfit is because increasing the model complexity increases the likelihood of overfitting. As a result, the total parameters of the model increased from ~30M to ~35M by adding just 4 additional layers.

The next change involved reducing the **input dataset size by fifty percent**. A small training data set can cause overfitting as it may not contain enough samples to represent all the potential data values fairly or accurately.

In addition, the Adam optimizer learning rate was adjusted to **0.01** from **0.0001**, and the number of epochs that training occurred for increased from a value of **15 to 20.**

Table 7: Performance Metrics with a Model that was Overfit



|**Performance Metric**|**Value**|
| - | - |
|Training Loss|0\.054248064452670365|
|Training Accuracy|0\.9843152257077277|
|Validation Loss|12151604578.59951|
|Validation Accuracy|0\.08951798010711554|
|Testing Accuracy|0\.09224069675376088|

Table 7 shows the performance metrics after the changes described above were implemented. Based on the poor validation and testing accuracy scores and the extremely high accuracy score for training, it’s clear that the model does not perform well on new data, a sign of overfitting. In addition, the stark contrast in the loss values between the training (i.e., low error value) and validation (i.e., extremely high value) is also an indicator that overfitting is occurring.

Therefore, it can be concluded that overfitting was successfully engineered and implemented based on the results displayed above.

<figure>
<img src="report images/overfit-model-acc.jpeg" alt="Image description">
  <figcaption>Figure 13: Observed training and validation loss and accuracy over epochs. The validation accuracy diverging from the training accuracy is a strong indication of the “overfitting” nature of the model.</figcaption>
</figure>


## 3. *Underfitting*

<a name="_page21_x72.00_y63.00"></a>The next step was to engineer underfitting purposefully. After careful research and deliberate testing, five main model elements were investigated and changed.

To start, the **model's complexity was significantly lowered**, which involved decreasing the number of depthwise convolutions and convolution layers. It's well known that decreasing the model's complexity increases the likelihood of underfitting. As a result, we ended up with a model with ~23M parameters compared to the original 30M parameters by removing 6 layers.

In addition, the next few changes involved **removing the Batch Normalization layer** and the **addition of dropout layers** (i.e., increased dropout rate). Batch normalization helps in stabilizing and accelerating the training process. Removing batch normalization layers can make training more challenging and lead to underfitting. Dropout randomly sets a fraction of input units to zero during training, which can prevent over-reliance on specific features., as a result, adding Dropout can contribute to the model’s hypothesis being underfitted.

Lastly, Instead of using advanced optimization algorithms like Adam, **switch to simpler optimizers like SGD** (Stochastic Gradient Descent) with momentum. This can slow down the convergence of the model, making it harder to fit the data. Additionally, setting a **high momentum rate** forces the model to accelerate convergence further leading to underfitting of the data.

Table 8: Performance Metrics with a Model that was Underfit



|**Performance Metric**|**Value**|
| - | - |
|Training Loss|3\.9223806462499735|
|Training Accuracy|0\.023265040097480802|
|Validation Loss|0\.03139905605244301|
|Validation Accuracy|0\.024744897959183672|
|Testing Accuracy|0\.02391132224861441|

Table 8 shows the performance metrics after the changes described above were implemented. Based on the poor validation, testing, and training accuracy scores, it’s clear that the model does not perform well on the data, a classic sign of underfitting. In addition, the small validation loss value implies that the loss function didn't converge, and the high training loss implies that the model made a lot of errors while classifying, two indicators that the model is underfitting. Therefore, it can be concluded that underfitting was successfully engineered and implemented based on the results displayed above.

# 10. **References**
1. <a name="_page22_x72.00_y123.83"></a>“GTSRB - German Traffic Sign Recognition Benchmark,” www.kaggle.com. [https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/data ](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/data)(accessed Mar. 14, 2024).
1. “Traffic sign classification and bbox model-PyTorch,” kaggle.com. [https://www.kaggle.com/code/divakaivan12/traffic-sign-classification-and-bbox-model-pytorch ](https://www.kaggle.com/code/divakaivan12/traffic-sign-classification-and-bbox-model-pytorch)(accessed Mar. 15, 2024).
1. A. G. Howard and M. Zhu, “MobileNets: Open-Source Models for Efficient On-Device Vision,” Google Research, Jun. 14, 2017. <https://blog.research.google/2017/06/mobilenets-open-source-models-for.html> (accessed Mar. 16, 2024).
1. A. G. Howard et al., “MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications,” Accessed: Mar. 16, 2024. [Online]. Available: <https://ar5iv.labs.arxiv.org/html/1704.04861>
1. “MobileNet from scratch,” kaggle.com. <https://www.kaggle.com/code/sonukiller99/mobilenet-from-scratch/notebook> (accessed Mar. 18, 2024).
1. Ł. Kaiser, G. Brain, A. Gomez, and F. Chollet, “Published as a conference paper at ICLR 2018 DEPTHWISE SEPARABLE CONVOLUTIONS FOR NEURAL MACHINE TRANSLATION.” Accessed: Mar. 18, 2024. [Online]. Available: <https://openreview.net/pdf?id=S1jBcueAb#:~:text=Notably%2C>
1. A. Sharma, “Cross Validation in Machine Learning,” GeeksforGeeks, Nov. 21, 2017. <https://www.geeksforgeeks.org/cross-validation-machine-learning/>
1. I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning. Cambridge, Massachusetts: The Mit Press, 2016.
1. Pytorch Neural Network Module by The Linux Foundation : <https://pytorch.org/docs/stable/nn.html>
