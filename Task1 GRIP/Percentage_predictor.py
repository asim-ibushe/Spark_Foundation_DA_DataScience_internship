'''
Regression :	Simple Linear Regression
DataSet :	Study_Score
Training Dataset :	study_hrs, Percentage obtained
Testing Dataset :	Actual study hours
'''

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

	
def main():
	print("----ML model to predict Student percentage using study Hrs per day----")
	#step1:Load data
	dataset=pd.read_csv("student_scores.csv")
	#print(dataset)
	print("size of dataframe:",dataset.shape)

	#step2:cleaning and analyse data	
	#independent variable
	X=dataset["Hours"].values
	X=X.reshape((-1,1))
	#dependent variable
	Y=dataset["Scores"].values
	data_train,data_test,target_train,target_test=train_test_split(X,Y,test_size=0.6)	
	print("Length of X values: ",len(X))
	print("Length of Y values: ",len(Y))
	Mean_X=np.mean(X)
	Mean_Y=np.mean(Y)
	print("Mean study hours(All student) is: ",Mean_X)
	print("Mean obtained marks(All student) is: ",Mean_Y)	

	#step3:Training phase
	#y=mx + c (slope of line)
	#select algorithm and train your model
	LR=LinearRegression()
	LR.fit(data_train,target_train)
	#calculating slope and y_intercept of Line of Regression 
	c=LR.intercept_
	print("Y-intercept= ",c)
	m=LR.coef_
	print("Slope of Line(m)= ",m)

	#calculating accuracy 
	y_predict=LR.predict(data_test)
	accuracy=r2_score(target_test,y_predict)*100
	print("Goodness of fit of Your LR line: ",accuracy,"%")
	if(accuracy>85):
		#step4:Test your ML model
		Hours=float(input("Enter Your Study Hours per day :"))
		Marks=float(LR.predict([[Hours]]))	
		print(f"\nPredicted Percentage for {Hours} Hrs/day of study using ML model is : {Marks:.2f}%\n")
	else:
		#step5:If Accuracy do not satisfy,retrain your model-provide another dataset.
		print("Re-train your model, Prediction might fail.Else try another ML model")

	X_start=np.min(X)
	X_end=np.max(X)
	x=np.linspace(X_start,X_end,len(X))
	y=m*x +c
	#plot your regression line and training data for better visualisation
	plt.plot(x,y,color='r',label='Regression Line')	
	plt.scatter(X,Y,color='b',label="Data Plot")
	
	plt.title("Study Hrs vs Obtained Percentage")
	plt.xlabel("Study Hours per day")
	plt.ylabel("Percentage of Student")
	plt.legend()
	plt.show() 
	
if __name__=="__main__":
	main()
