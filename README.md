# Apply CNN to gait recognition based on inertial signals  
My research focused on gait recognition in machine learning lab of Wuhan University, aiming to collect tri-axial accelerometer and gyroscope signals, performing processing and feature extraction, developing CNN model for identification. The accuracy of the optimal model on the out-of-sample test reached 93.4%.  
### Step 1: Collecting tri-axial accelerometer and gyroscope signals of walking
![image](https://github.com/Luffy-wu/picture/blob/master/图片%201.png)      
### Step 2: Preprocessing and feature extraction  
•	Removed unstable signals at the beginning and end of walking  
•	Segmented data into 128 lines  
•	Normalized data and filtered data noise using median filters  
•	Matched it with one-hot data tags, then randomly shuffled it, divided it into training & test set    
![image](https://github.com/Luffy-wu/picture/blob/master/图片%202.png)    
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Input wave and filtered wave 
### Step 3: Constructing CNN model  
Weight initialization-- convolution and pooling--dense layer--dropout--output layer--training and evaluating.   
Especially, used Relu in linear layers and sigmoid function in output layer and set up the batch set to randomly select a small subset for training and used Adam optimizer to accelerate training.  
![image](https://github.com/Luffy-wu/picture/blob/master/图片%203.png)   
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Network structure of constructed CNN model  
![image](https://github.com/Luffy-wu/picture/blob/master/图片%204.png)   
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;The shuffled data of acc_x  
In addition, respectively set up a one-channel, three-channel, and six-channel input layer model to compare the impact of different input structure on the model's performance.  
![image](https://github.com/Luffy-wu/picture/blob/master/图片%205.png)   
### Step 4: Train the model  
Research experiments found that the accuracy of the optimal model on out-of-sample test reached 93.4%, indicating that CNN model can well identify gait characteristics based on inertial signals. Meanwhile, it found that the standardized data can exclude noise and outliers to a certain extent. Compared with unstandardized raw data, it can significantly improve recognition accuracy. In addition, the longitudinal joint of accelerometer and gyroscope signals   in X, Y, and Z axes can achieve effective fusion of features in three directions, and further enhance recognition accuracy. Through comparison among one-channel, three-channel, and six-channel models, it was found that three channels can significantly improve accuracy, but the further promotion effect of six channels is minimal.  
![image](https://github.com/Luffy-wu/picture/blob/master/图片%206.png)
![image](https://github.com/Luffy-wu/picture/blob/master/图片%207.png)
