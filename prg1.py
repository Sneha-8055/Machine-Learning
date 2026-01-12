# from sklearn.datasets import make_moons
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# # Create synthetic 2D data
# X, y = make_moons(n_samples=300, noise=0.3, random_state=42)
# # Create a DataFrame for plotting
# df = pd.DataFrame(X, columns=["Sneha", "Aparna"])
# df['Target'] = y
# # Visualize the 2D data
# plt.figure(figsize=(8, 6))
# sns.scatterplot(data=df, x="Sneha", y="Aparna", hue="Target", palette="Set1")
# plt.title("visualising 2d data)")
# plt.grid(True)
# plt.show()

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# # Normalize the features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# # Split into train and test
# X_train, X_test, y_train, y_test = train_test_split(
# X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score
# # Train a k-NN classifier
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train, y_train)
# # Predict and evaluate
# y_pred = knn.predict(X_test)
# print(f"Test Accuracy (k=5): {accuracy_score(y_test, y_pred):.2f}")

# from sklearn.model_selection import cross_val_score
# import numpy as np
# # Range of k values to try
# k_range = range(1, 21)
# cv_scores = []
# # Evaluate each k using 5-fold cross-validation
# for k in k_range:
#  knn = KNeighborsClassifier(n_neighbors=k)
#  scores = cross_val_score(knn, X_scaled, y, cv=5, scoring='accuracy')
#  cv_scores.append(scores.mean())
# # Plot accuracy vs. k
# plt.figure(figsize=(8, 5))
# plt.plot(k_range, cv_scores, marker='o')
# plt.title("k-NN Cross-Validation Accuracy vs k")
# plt.xlabel("Number of Neighbors: k")
# plt.ylabel("Cross-Validated Accuracy")
# plt.grid(True)
# plt.show()
# # Best k
# best_k = k_range[np.argmax(cv_scores)]
# print(f"Best k from cross-validation: {best_k}")

# # Train final model with best k
# best_knn = KNeighborsClassifier(n_neighbors=best_k)
# best_knn.fit(X_train, y_train)
# # Predict on test data
# y_pred = best_knn.predict(X_test)

# from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
# # Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])
# disp.plot(cmap="Blues")
# plt.title(f"Confusion Matrix (k={best_k})")
# plt.grid(False)
# plt.show()
# # Detailed classification report
# print("Classification Report:")
# print(classification_report(y_test, y_pred, target_names=["Class 0", "Class 1"]))

# # Predict on mesh grid with best k
# # Create mesh grid
# from sklearn.neighbors import KNeighborsClassifier

# k = 8  #  give k value here

# knn = KNeighborsClassifier(n_neighbors=k)
# knn.fit(X_train, y_train)

# best_knn = knn   # use this for plotting
# best_k = k
# x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
# y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1

# xx, yy = np.meshgrid(
#     np.arange(x_min, x_max, 0.02),
#     np.arange(y_min, y_max, 0.02)
# )
# Z = best_knn.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
# # Plot decision boundary
# plt.figure(figsize=(8, 6))
# plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.4)
# sns.scatterplot(x=X_scaled[:, 1], y=X_scaled[:, 0], hue=y, palette="Set2", edgecolor='k')
# plt.title(f"Decision Boundary with Best k = {best_k}")
# plt.xlabel("Feature 1 (scaled)")
# plt.ylabel("Feature 2 (scaled)")
# plt.grid(True)
# plt.show()


# import pandas as pd
# import numpy as np
# data = {
# 'School ID': [101, 102, 103, np.nan, 105, 106, 107, 108],
# 'Name': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
# 'Address': ['123 Main St', '456 Oak Ave', '789 Pine Ln', '101 Elm St', np.nan, '222 Maple Rd', '444 Cedar Blvd', '555 Birch Dr'],
# 'City': ['Mumbai', 'Delhi', 'Bengaluru', 'Chennai', 'Kolkata', np.nan, 'Pune', 'Jaipur'],
# 'Subject': ['Math', 'English', 'Science', 'Math', 'History', 'Math', 'Science', 'English'],
# 'Marks': [85, 92, 78, 89, np.nan, 95, 80, 88],
# 'Rank': [2, 1, 4, 3, 8, 1, 5, 3],
# 'Grade': ['B', 'A', 'C', 'B', 'D', 'A', 'C', 'B']
# }
# df = pd.DataFrame(data)
# print("Sample DataFrame:")
# print(df)

# df_cleaned = df.dropna()
# print("\nDataFrame after removing rows with missing values:")
# print(df_cleaned)

# mean_imputation = df['Marks'].fillna(df['Marks'].mean())
# median_imputation = df['Marks'].fillna(df['Marks'].median())
# mode_imputation = df['Marks'].fillna(df['Marks'].mode().iloc[0])
# print("\nImputation using Mean:")
# print(mean_imputation)
# print("\nImputation using Median:")
# print(median_imputation)
# print("\nImputation using Mode:")
# print(mode_imputation)

# forward_fill = df['Marks'].fillna(method='ffill')
# backward_fill = df['Marks'].fillna(method='bfill')
# print("\nForward Fill:")
# print(forward_fill)
# print("\nBackward Fill:")
# print(backward_fill)

# linear_interpolation = df['Marks'].interpolate(method='linear')
# quadratic_interpolation = df['Marks'].interpolate(method='quadratic')
# print("\nLinear Interpolation:")
# print(linear_interpolation)
# print("\nQuadratic Interpolation:")
# print(quadratic_interpolation)


# # Python code for Feature Scaling using Robust Scaling
# """ PART 1: Importing Libraries """
# import pandas as pd
# import numpy as np
# from sklearn import preprocessing
# import matplotlib
# import matplotlib.pyplot as plt
# import seaborn as sns  
# matplotlib.style.use('ggplot')
# """ PART 2: Making the data distributions """
# x = pd.DataFrame({

#  # Distribution with lower outliers
#  'x1': np.concatenate([np.random.normal(30, 1, 200), np.random.normal(1, 2, 200)]),
#  # Distribution with higher outliers
#  'x2': np.concatenate([np.random.normal(10, 1, 200), np.random.normal(80, 2, 200)]),
# })
# """ PART 3: Scaling the Data """
# scaler = preprocessing.RobustScaler()
# robust_scaled_df = scaler.fit_transform(x)
# robust_scaled_df = pd.DataFrame(robust_scaled_df, columns=['x1', 'x2'])
# """ PART 4: Visualizing the impact of scaling """
# fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(9, 7))
# ax1.set_title('Before Scaling')
# sns.kdeplot(x['x1'], ax=ax1)
# sns.kdeplot(x['x2'], ax=ax1)
# ax2.set_title('After Robust Scaling')
# sns.kdeplot(robust_scaled_df['x1'], ax=ax2)
# sns.kdeplot(robust_scaled_df['x2'], ax=ax2)
# plt.show()


# import pandas as pd
# import numpy as np
# from sklearn import preprocessing
# import matplotlib
# import matplotlib.pyplot as plt
# import seaborn as sns

# matplotlib.style.use('fivethirtyeight')
# x = pd.DataFrame({
#  'x1': np.concatenate([np.random.normal(20, 2, 1000), np.random.normal(1, 2, 25)]),
#  'x2': np.concatenate([np.random.normal(30, 2, 1000), np.random.normal(50, 2, 25)]),
# })
# np.random.normal
# scaler = preprocessing.RobustScaler()
# robust_df = scaler.fit_transform(x)
# robust_df = pd.DataFrame(robust_df, columns=['x1', 'x2'])
# scaler = preprocessing.StandardScaler()
# standard_df = scaler.fit_transform(x)
# standard_df = pd.DataFrame(standard_df, columns=['x1', 'x2'])
# scaler = preprocessing.MinMaxScaler()
# minmax_df = scaler.fit_transform(x)
# minmax_df = pd.DataFrame(minmax_df, columns=['x1', 'x2'])
# fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(20, 5))
# ax1.set_title('Before Scaling')
# sns.kdeplot(x['x1'], ax=ax1, color='r')
# sns.kdeplot(x['x2'], ax=ax1, color='b')
# ax2.set_title('After Robust Scaling')
# sns.kdeplot(robust_df['x1'], ax=ax2, color='red')
# sns.kdeplot(robust_df['x2'], ax=ax2, color='blue')
# ax3.set_title('After Standard Scaling')
# sns.kdeplot(standard_df['x1'], ax=ax3, color='black')
# sns.kdeplot(standard_df['x2'], ax=ax3, color='g')
# ax4.set_title('After Min-Max Scaling')
# sns.kdeplot(minmax_df['x1'], ax=ax4, color='black')
# sns.kdeplot(minmax_df['x2'], ax=ax4, color='g')
# plt.show()


# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# df = pd.read_csv('heart.csv')
# X = df.drop('target', axis=1)
# y = df['target']
# df.head()
# features = ['age','trestbps','chol','thalach','oldpeak']
# scaler = MinMaxScaler()
# X_normalized = X.copy()
# X_normalized[features] = scaler.fit_transform(X[features])
# X_normalized.head()
# scaler_z = StandardScaler()
# X_standardized = X.copy()
# X_standardized[features] = scaler_z.fit_transform(X[features])
# X_standardized.head()

# import pandas as pd

# # Read CSV from same folder as your .py file
# df = pd.read_csv('heart.csv')

# X = df.drop('target', axis=1)
# y = df['target']

# print(df.head())

# from sklearn.preprocessing import MinMaxScaler

# # Features to normalize
# features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# # Initialize scaler
# scaler = MinMaxScaler()

# # Copy original data
# X_normalized = X.copy()

# # Apply Min-Max scaling
# X_normalized[features] = scaler.fit_transform(X[features])

# # View result
# X_normalized.head()

# import pandas as pd

# # Read CSV from same folder as your .py file
# df = pd.read_csv('heart.csv')

# X = df.drop('target', axis=1)
# y = df['target']

# print(df.head())

# from sklearn.preprocessing import MinMaxScaler

# # Features to normalize
# features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# # Initialize scaler
# scaler = MinMaxScaler()

# # Copy original data
# X_normalized = X.copy()

# # Apply Min-Max scaling
# X_normalized[features] = scaler.fit_transform(X[features])

# # View result
# X_normalized.head()

# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import GridSearchCV
# import numpy as np 
# from sklearn.datasets import make_classification

# x,y=make_classification(
#     n_samples=1000,n_features=40,n_informative=20,n_classes=4,random_state=32)

# c_space=np.logspace(-6,9,20)
# param_grid={'C':c_space}
# logreg=LogisticRegression()
# logreg_cv=GridSearchCV(logreg,param_grid,cv=8)
# logreg_cv.fit(x,y)
# print("Tuned logistic regresssion parameters:()".format(logreg_cv.best_params_))
# print("Best score is{}".format(logreg_cv.best_score_))


# import numpy as np
# from sklearn.datasets import make_classification
# x,y=make_classification(n_samples=1000,n_features=20,n_informative=10,n_classes=2,random_state=42)
# from scipy.stats import randint
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import RandomizedSearchCV

# param_dist={
#     "max_depth":[5,None],
#     "max_features":randint(1,90),
#     "min_samples_leaf":randint(1,90),
#     "criterion":["gini","entropy"]

# }
# tree =DecisionTreeClassifier()
# tree_cv=RandomizedSearchCV(tree,param_dist,cv=7)
# tree_cv.fit(x,y)

# print("Tuned Decision tree parameters:{}".format(tree_cv.best_params_))
# print("Best score is{}".format(tree_cv.best_score_))



# from sklearn.metrics import (
#     confusion_matrix,
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     roc_curve,
#     auc
#      )
# y_true=[0,1,0,0,1,0,1,1,1,0]
# y_pred=[0,1,0,0,1,1,0,1,1,1]
# cm=confusion_matrix(y_true,y_pred)
# accuracy=accuracy_score(y_true,y_pred)
# precision=precision_score(y_true,y_pred)
# recall=recall_score(y_true,y_pred)
# f1=f1_score(y_true,y_pred)
# fpr,tpr,thresholds=roc_curve(y_true,y_pred)
# roc_auc=auc(fpr,tpr)
# print("confusion matrix:",cm)
# print("accuracy score:",accuracy)
# print("precision score:",precision)
# print("recall score:",recall)
# print("f1 score:",f1)
# print("roc auc:",roc_auc)



# import matplotlib.pyplot as plt
# # Plot ROC curve
# plt.figure()
# # Plot the ROC curve with a label displaying the ROC AUC score
# plt.plot(fpr, tpr, color='darkorange', lw=2,
#  label='ROC curve (area = %0.2f)' % roc_auc)
# # Plot a dashed diagonal line for reference
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# # Set the x and y-axis limits
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# # Label the x and y-axes
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# # Set the title of the plot
# plt.title('Receiver Operating Characteristic')
# # Add a legend to the plot
# plt.legend(loc='lower right')
# # Display the ROC curve plot
# plt.show()

# import matplotlib.pyplot as plt

# plt.figure()
# plt.plot(
#     fpr,
#     tpr,
#     color='darkorange',
#     lw=2,
#     label='ROC curve (area=%0.2f)' roc_auc
# )
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])

# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')

# plt.legend(loc='lower right')
# plt.show()
