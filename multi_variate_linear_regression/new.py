import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class LinearRegression:
    
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self,X_train,y_train):
        
        #adding ones in the place of the first column
        newX_train = np.insert(X_train,0,1,axis=1)

        #calculating coefficients:
        parameters = np.linalg.inv(np.dot(newX_train.T,newX_train)).dot(newX_train.T).dot(Y_train)
        self.intercept_= parameters[0]
        self.coef_ = parameters[1:]
        #print(parameters)

    def predict(self,X_test):
        y_pred = np.dot(X_test,self.coef_) + self.intercept_
        return y_pred
    
    

data = pd.read_csv('multiple_linear_regression_dataset.csv')
lr = LinearRegression()

X_train = data[['age','experience']]
Y_train = data['income']

X_test = [33,2]

lr.fit(X_train,Y_train)
print("salary = ",lr.predict(X_test))


#<----for graph start----->
# Extract attributes
age = data['age']
experience = data['experience']
income = data['income']

# Create a new figure
fig = plt.figure()

# Add a 3D subplot
ax = fig.add_subplot(111, projection='3d')

# Plot a scatter plot
ax.scatter(age, experience, income, c='b', marker='o')

# Add labels
ax.set_xlabel('Age')
ax.set_ylabel('Experience')
ax.set_zlabel('Income')

# Add a title
ax.set_title('3D Scatter Plot of Age, Experience, and Income')


# Define the coefficients
a = lr.coef_[0]
b = lr.coef_[1]
c = lr.intercept_

# Create a grid of x and y values
x = np.linspace(age.min(), age.max(), 100)
y = np.linspace(experience.min(), experience.max(), 100)
x, y = np.meshgrid(x, y)

# Compute the z values using the plane equation
z = a * x + b * y + c

# Plot the plane
ax.plot_surface(x, y, z, color='r', alpha=0.5, rstride=100, cstride=100)

# Show the plot
plt.show()


#<----for graph end----->


