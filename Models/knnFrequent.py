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
most_frequent_values = df2.mode().iloc[0]
df2_filled = df2.fillna(most_frequent_values)



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

print("Missing Values Before Handling:")
print(df.isnull().sum())

   
print("\nMissing Values After Handling:")





# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import joblib


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy="most_frequent") 
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}

knn_classifier = KNeighborsClassifier()

grid_search = GridSearchCV(knn_classifier, param_grid, cv=5, scoring='accuracy')

grid_search.fit(X_train_imputed, y_train)
best_k = grid_search.best_params_['n_neighbors']

knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train_imputed, y_train)

knn_predictions = knn_model.predict(X_test_imputed)

accuracy = accuracy_score(y_test, knn_predictions)
classification_rep = classification_report(y_test, knn_predictions)
print("Best k value:", best_k)
print("KNN Accuracy (with imputed data):", accuracy)
print("Classification Report:")
print(classification_rep)
joblib.dump(knn_model, 'knnFrequent.pkl')


# %% [markdown]
# 

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

joblib.dump(knn_model, 'knnFrequent.pkl')

# Display results
print("KNN Accuracy:", accuracy)
print("Classification Report:")
print(classification_rep)

# %%
df.columns

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



