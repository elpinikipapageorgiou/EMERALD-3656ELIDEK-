# Neural-FCM

## Abstarct

Neural-FCM is a novel learning framework that integrates deep learning concepts into Fuzzy Cognitive Map (FCM) modeling to improve classification performance. By employing a hybrid artificial neural network (ANN) decoder architecture, Neural-FCM transforms input data instances into FCM weight matrices. The method leverages deep learning optimization techniques, incorporating FCM reasoning into the training process by embedding it within a differentiable loss function. This approach ensures that the network outputs weight matrices that enhance FCM inference accuracy, while maintaining the fundamental interpretability of FCMs. Neural-FCM's dynamic weight generation allows for instance-specific matrix outputs, similar to Fuzzy Grey Cognitive Maps, enabling the model to adapt effectively to input data and deliver robust classification results. The use of high-level tools like Python and TensorFlow supports the practicality and reproducibility of the method, fostering further research and application in various domains. 

**Neural-FCM is currently under evaluation for publication in a scientific journal with open-access. Once it is published, the training scripts, as well as the complete implementation details along with the paper reference will be included in this repository.** 

### How to use this project

To use this project you must to install the packages of **requirements.txt** file and having a **python 3.10 version (recommended)** installed in your machine.


The python script *inference.py* provides the code for loading the Neural-FCM model and the CAD sub-dataset. After loading, the data preparation is performed, with the corresponding preproccesing modules being provided. Finally, FCM inference is performed based on the model parameters, on the CAD dataset using the Neural-FCM models. 

For more information you should contact Dr. Elpiniki Papageorgiou or the corresponding author Theodoros Tziolas <ttziolas@uth.gr>

