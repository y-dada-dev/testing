# %% [markdown]
# Dataset: Use the famous Iris dataset, which is available in many machine learning libraries. This dataset consists of 150 samples of iris flowers, each belonging to one of three species: setosa, versicolor, or virginica.

# %%
import pandas as pd

file_path1 = './clean.xlsx'
file_path2 = './trash.xlsx'

df1 = pd.read_excel(file_path1)
df2 = pd.read_excel(file_path2)

new_column_names_dataset1 = {'area': 'Area', 'perimeter': 'Perimeter', 'MajorAL': 'MajorAxisLength', 'MinorAL': 'MinorAxisLength', 'AspectRation': 'AspectRation', 'Eccentricity': 'Eccentricity', 'ConvexArea': 'ConvexArea', 'EquivDiameter': 'EquivDiameter', 'Extent': 'Extent', 'Solidity': 'Solidity', 'Roundness': 'Roundness', 'compactness': 'Compactness', 'shapeFactor1': 'ShapeFactor1', 'shapeFactor2': 'ShapeFactor2', 'shapeFactor3': 'ShapeFactor3', 'shapeFactor4': 'ShapeFactor4', 'Class': 'Class'}

df2 = df2.rename(columns=new_column_names_dataset1)


df2_filled = df2.dropna()
df = pd.concat([df1, df2_filled], ignore_index=True)

print("\nDataset Statistics:")
print(df.describe())

# %% [markdown]
# This code creates a pair plot to visualize relationships between features and a heatmap of the correlation matrix. Adjust the column names based on your actual dataset structure.

# %%
import matplotlib.pyplot as plt
import seaborn as sns

subset_df = df.head(100)

# Pair plot to visualize relationships between features
sns.pairplot(subset_df, hue='Class')  # Assuming 'class' is the column containing the class labels
plt.show()

# Correlation matrix heatmap
correlation_matrix = subset_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


# Assuming 'cleaned_df' is your DataFrame
# Features (X) and target variable (y)
X = df.drop("Class", axis=1)
y = df["Class"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# %%
print(X)

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Assuming X and y are your feature matrix and target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for GridSearchCV
param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}  # Adjust the list of k values as needed

# Create KNN classifier
knn_classifier = KNeighborsClassifier()

# Create GridSearchCV
grid_search = GridSearchCV(knn_classifier, param_grid, cv=5, scoring='accuracy')

# Fit the model with the best hyperparameters
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_k = grid_search.best_params_['n_neighbors']

# Create KNN classifier with the best k
knn_model = KNeighborsClassifier(n_neighbors=best_k)

# Train the KNN model
knn_model.fit(X_train, y_train)

# Predictions on the test set
knn_predictions = knn_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, knn_predictions)
classification_rep = classification_report(y_test, knn_predictions)

# Display results
print("Best k value:", best_k)
print("KNN Accuracy:", accuracy)
print("Classification Report:")
print(classification_rep)


# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


# Create KNN classifier with k=3
knn_model = KNeighborsClassifier(n_neighbors=7)

# Train the KNN model
knn_model.fit(X_train, y_train)

# Predictions on the test set
knn_predictions = knn_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, knn_predictions)
classification_rep = classification_report(y_test, knn_predictions)

# Display results
print("KNN Accuracy:", accuracy)
print("Classification Report:")
print(classification_rep)
joblib.dump(knn_model, 'knnDropped.pkl')


# %%


# Assuming cleaned_df is your DataFrame and X_clustering is your features
# Change 'Cluster' to the actual column name if it's different
attributes = ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRation',
              'Eccentricity', 'ConvexArea','EquivDiameter', 'Extent', 'Solidity', 'roundness',
              'Compactness', 'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4']

# Create scatter plots for each pair of attributes
for i in range(len(attributes)):
    for j in range(i + 1, len(attributes)):
        attribute1 = attributes[i]
        attribute2 = attributes[j]

        plt.scatter(X[attribute1], X[attribute2], cmap='viridis', s=50)
        plt.title(f'K-Means Clustering - {attribute1} vs {attribute2}')
        plt.xlabel(attributes[i])
        plt.ylabel(attributes[j])
        plt.show()



