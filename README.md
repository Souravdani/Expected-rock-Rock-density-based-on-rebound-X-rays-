# Expected-rock-Rock-density-based-on-rebound-X-rays-
A generalized model which can be used to accept rebound signal and output the expected rock density.
We are making a tunnel in which according to different rocks we are using different machine boring heads
We have experimental X-ray signal data used to determine the rock density.
Hence our goal is to buid a generalized model; which can be usedto accept a rbound signal and predict the expected rock density.
With th ehelp of predicted rock density, we can decide for which boring head to be used.


##### In this project, we are exploring different machine learning models:
1- Linear Regression
2- Polynomial regression
3- KNN regression
4- Decision tree regresssion
5- Support vector regression
6- Boosted trees regression
7- Random forest regression

###  Reviews ###
The linear regression had anomly in predicted values( all values around 2.2) altough the mean absolute error and mean squared error looked fine.
The reason was that the model was heavily underfitted. WE could visualize it by plotting the predicted values.
Other model performed well on the training data but were overfitting our data.
The best model output was seen from Boosted trees model 
hence we take boosted tree model as our final model. 
