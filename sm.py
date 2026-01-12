import keyword
print("the list of keywords are:")
print(keyword.kwlist)

age_input=input("enter your age:")
age=int(age_input)
if age<0:
    print("enter the valid age.")
elif age<18:
    print("you are a minor")
elif age>=18 and age<65:
    print("you are a adult")
else:
    print("you are a senior citizen")
    
    
    for i in range(1,11):
        if i==6:
         continue
    else:
        print(i,end=" ")
        
        
expr=10+25*35
print(expr)

x=15
print("x is of type:",type(x))
y=10.5
print("y is of type:",type(y))
z=x+y
print(z)
print("z is of type:",type(z))

s="10010"
c=int(s,2)
print("after converting into integer base 2:",end="")
print(c)
e=float(s)
print("after converting to float:",end=" ")
print(e)

a=[1,2,3,4,5]
b=['apple','banana','cherry']
c=[1,'hello',3.14,True]
print(a)
print(b)
print(c)

t=(10,5,20)
print("value in t[0]=",t[0])
print("value in t[-1]=",t[-1])
print("value in t[-2]=",t[-2])

t=('savita',)*10
print(t)

#t=(0,1,2,3,4)
#print(t[2:])
#print(t[::-2])
#print(t[3:4])

# t=(0,1)
# del t
# print(t)


a = [10, 20, 30, 40]

a.remove(20)   
print(a)      

a.pop(-1)      
print(a)       
import os

def create_file(filename):
    try:
        with open(filename, 'w') as f:
            f.write('Certification in Advanced Machine Learning and Introductory Deep Learning — From Foundations to Practice_Clg_Syllabus')
        print("File " + filename + " created successfully.")
    except IOError:
        print("Error: could not create file " + filename)

def read_file(filename):
    try:
        with open(filename, 'r') as f:
            contents = f.read()
            print(contents)
    except IOError:
        print("Error: could not read file " + filename)

def append_file(filename, text):
    try:
        with open(filename, 'a') as f:
            f.write(text)
        print("Text appended to file " + filename + " successfully.")
    except IOError:
        print("Error: could not append to file " + filename)

def rename_file(filename, new_filename):
    try:
        os.rename(filename, new_filename)
        print("File " + filename + " renamed to " + 
                  new_filename + " successfully.")
    except IOError:
        print("Error: could not rename file " + filename)

def delete_file(filename):
    try:
        os.remove(filename)
        print("File " + filename + " deleted successfully.")
    except IOError:
        print("Error: could not delete file " + filename)


if __name__ =='__main__':
    filename = "example.txt"
    new_filename = "new_example.txt"

    create_file(filename)
    read_file(filename)
    append_file(filename, "Python provides built-in functions to handle file operations such as reading from and writing to files. We can use the open() function to work with files.\n")
    read_file(filename)
    rename_file(filename, new_filename)
  #  read_file(new_filename)
   # delete_file(new_filename)
   
# import math
# print("the value of log 2 with base 3 is:",end="")
# print(math.log(2,3))
# print("the value of log2 of 16 is:",end="")
# print(math.log2(16))

# importing the csv module
import csv
# my data rows as dictionary objects
mydict =[{'branch': 'COE', 'cgpa': '9.0', 'name': 'Nikhil', 'year': '2'},
{'branch': 'COE', 'cgpa': '9.1', 'name': 'Sanchit', 'year': '2'},
{'branch': 'IT', 'cgpa': '9.3', 'name': 'Aditya', 'year': '2'},
{'branch': 'SE', 'cgpa': '9.5', 'name': 'Sagar', 'year': '1'},
{'branch': 'MCE', 'cgpa': '7.8', 'name': 'Prateek', 'year': '3'},
{'branch': 'EP', 'cgpa': '9.1', 'name': 'Sahil', 'year': '2'}]
# field names
fields = ['name', 'branch', 'year', 'cgpa']
# name of csv file
filename = "university_records.csv"
# writing to csv file
with open(filename, 'w') as csvfile:
   writer = csv.DictWriter(csvfile, fieldnames = fields)
# writing headers (field names)
   writer.writeheader()
# writing data rows
   writer.writerows(mydict)
   
import math
a=math.pi/6
print("the value of sine of pi/6 is:",end="")
print(math.sin(a))
print("the value of cosine of pi/6 is:",end="")
print(math.cos(a))
print("the value of tangent of pi/6 is:",end="")
print(math.tan(a))

import pandas as pd

df = pd.read_csv(r"D:\pi164\data.csv")
print(df.head())

import numpy as np
import time
SIZE=100000
L1=range(SIZE)
L2=range(SIZE)
A1=np.arange(SIZE)
A2=np.arange(SIZE)
start=time.time()
result=[(x,y)for x,y in zip(L1,L2)]
print((time.time()-start)*1000)
start=time.time()
result=A1+A2
print((time.time()-start)*1000)

import pandas as pd
import numpy as np
dict = {'first score':[100,90,np.nan,95],
        'second score':[30,45,56,np.nan],
        'third score':[np.nan, 40,80,98]}
df=pd.DataFrame(dict)
print(df.isnull())

from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC
cancer = load_breast_cancer()
X = cancer.data[:, :2]
y = cancer.target
svm = SVC(kernel="linear", C=1)
svm.fit(X, y)
DecisionBoundaryDisplay.from_estimator(
svm,
X,
response_method="predict",
alpha=0.8,
cmap="Pastel1",
xlabel=cancer.feature_names[0],
ylabel=cancer.feature_names[1],
)
plt.scatter(X[:, 0], X[:, 1], 
c=y, 
s=20, edgecolors="black")

plt.title("SVM Decision Boundary with Linear", fontweight="bold",color="Red")
plt.show()

import numpy as np
from collections import Counter           

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))

def knn_predict(training_data, training_labels, test_point, k):
    distances = []
    for i in range(len(training_data)):
        dist = euclidean_distance(test_point, training_data[i])
        distances.append((dist, training_labels[i]))
        distances.sort(key=lambda x: x[0])
        k_nearest_labels = [label for _, label in distances[:k]]
    return Counter(k_nearest_labels).most_common(1)[0][0]
    sneha
training_data = [[1, 2], [2, 3], [3, 4], [6, 7], [7, 8]]
training_labels = ['A', 'A', 'A', 'B', 'B']
test_point = [4, 5]
k = 3

prediction = knn_predict(training_data, training_labels, test_point, k)
print(prediction)

import numpy as np
X_grid = np.arange(min(X[:, 0]), max(X[:, 0]), 0.01)  # Only the first feature
X_grid = X_grid.reshape(-1, 1)
X_grid = np.hstack((X_grid, np.zeros((X_grid.shape[0], 2))))  # Pad with zeros
plt.scatter(X[:, 0], y, color='blue', label="Actual Data")
plt.plot(X_grid[:, 0], regressor.predict(X_grid), color='green', label="Random Forest Prediction")  
plt.title("Random Forest Regression Results")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend()
plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
actual = [0, 1, 1, 0, 1, 0, 1, 1]
predicted = [0, 1, 0, 0, 1, 0, 1, 0]
precision = precision_score(actual, predicted)
recall = recall_score(actual, predicted)
f1 = f1_score(actual, predicted)
metrics = ['Precision', 'Recall', 'F1 Score']
values = [precision, recall, f1]
plt.figure(figsize=(6, 4))
plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'salmon'])
plt.ylim(0, 1)
plt.title('Precision, Recall, and F1 Score')
plt.ylabel('Score')
for i, v in enumerate(values):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
plt.show()

import speech_recognition as sr

# Initialize recognizer class (for recognizing the speech)
r = sr.Recognizer()

# Reading Microphone as source
# listening the speech and store in audio_text variable
with sr.Microphone() as source:
    print("Talk")
    audio_text = r.listen(source)
    print("Time over, thanks")
    # recoginze_() method will throw a request
    # error if the API is unreachable,
    # hence using exception handling
    
    try:
        # using google speech recognition
        print("Text: "+r.recognize_google(audio_text))
    except:
         print("Sorry, I did not get that")
         
         
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import warnings

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')

df= pd.read_csv('Position_Salaries.csv')
print(df)

df.info()

X = df.iloc[:,1:2].values
y = df.iloc[:,2].values

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
x_categorical = df.select_dtypes(include=['object']).apply(label_encoder.fit_transform)
x_numerical = df.select_dtypes(exclude=['object']).values
x = pd.concat([pd.DataFrame(x_numerical), x_categorical], axis=1).values

regressor = RandomForestRegressor(n_estimators=28, random_state=10, oob_score=True)

regressor.fit(x, y)

from sklearn.metrics import mean_squared_error, r2_score

oob_score = regressor.oob_score_
print(f'Out-of-Bag Score: {oob_score}')

predictions = regressor.predict(x)

mse = mean_squared_error(y, predictions)
print(f'Mean Squared Error: {mse}')

r2 = r2_score(y, predictions)
print(f'R-squared: {r2}')

import numpy as np
   
X_grid = np.arange(min(X[:, 0]), max(X[:, 0]), 0.01)  # Only the first feature
X_grid = X_grid.reshape(-1, 1)
X_grid = np.hstack((X_grid, np.zeros((X_grid.shape[0], 2))))  # Pad with zeros

plt.scatter(X[:, 0], y, color='blue', label="Actual Data")
plt.plot(X_grid[:, 0], regressor.predict(X_grid), color='green', label="Random Forest Prediction")  
plt.title("Random Forest Regression Results")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend()
plt.show()

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

tree_to_plot = regressor.estimators_[0]
 
plt.figure(figsize=(20, 10))
plot_tree(tree_to_plot, feature_names=df.columns.tolist(), filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree from Random Forest")
plt.show()

