# %% [markdown]
# Dataset: Use the famous Iris dataset, which is available in many machine learning libraries. This dataset consists of 150 samples of iris flowers, each belonging to one of three species: setosa, versicolor, or virginica.

# %%
import pandas as pd

file_path1 = './clean.xlsx'
file_path2 = './trash.xlsx'



# Read the Excel file into a DataFrame
df1 = pd.read_excel(file_path1)
df2 = pd.read_excel(file_path2)


# Working With dataset2 
new_column_names_dataset1 = {'area': 'Area', 'perimeter': 'Perimeter', 'MajorAL': 'MajorAxisLength', 'MinorAL': 'MinorAxisLength', 'AspectRation': 'AspectRation', 'Eccentricity': 'Eccentricity', 'ConvexArea': 'ConvexArea', 'EquivDiameter': 'EquivDiameter', 'Extent': 'Extent', 'Solidity': 'Solidity', 'Roundness': 'Roundness', 'compactness': 'Compactness', 'shapeFactor1': 'ShapeFactor1', 'shapeFactor2': 'ShapeFactor2', 'shapeFactor3': 'ShapeFactor3', 'shapeFactor4': 'ShapeFactor4', 'Class': 'Class'}


df2 = df2.rename(columns=new_column_names_dataset1)
df2_filled = df2.dropna()



df = pd.concat([df1, df2_filled], ignore_index=True)
# df.to_excel('./df.xlsx', index=False)


# Display basic statistics about the dataset

print("\nDataset Statistics:")
print(df.head(100))


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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Assuming 'cleaned_df' is your DataFrame
# Features (X) and target variable (y)
X = df.drop("Class", axis=1)
y = df["Class"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#X.sample(frac=0)


# %%
print(len(X_train))

# %% [markdown]
# k-means

# %%
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score, make_scorer
import pandas as pd
import numpy as np
import joblib

# Assuming 'cleaned_df' is your DataFrame
X_clustering = df.drop("Class", axis=1)

# Specify the range of k values to search over
param_grid = {'n_clusters': range(2, 11)}  # Adjust the range as needed

# Create a KMeans instance
kmeans = KMeans(random_state=42)

# Create silhouette scorer
silhouette_scorer = make_scorer(silhouette_score)

# Create GridSearchCV instance
grid_search = GridSearchCV(kmeans, param_grid, scoring=silhouette_scorer, cv=5)

# Fit the model to the data
grid_search.fit(X_clustering)

# Check if the best score is NaN
if np.isnan(grid_search.best_score_):
    print("Silhouette Score is NaN. Try adjusting the range of k values or preprocessing your data.")
else:
    # Get the best k value from the grid search
    best_k = grid_search.best_params_['n_clusters']

    print(f"Best number of clusters (k): {best_k}")

    # Access the silhouette score for the best k
    best_silhouette_score = grid_search.best_score_
    print(f"Best Silhouette Score: {best_silhouette_score}")

    # Visualize the clusters with the best k
    best_kmeans = KMeans(n_clusters=best_k, random_state=42)
    best_kmeans.fit(X_clustering)
    df['Best_Cluster'] = best_kmeans.labels_
    joblib.dump(best_kmeans, "kmeansDropped.pkl")


    # Visualize the clusters
    plt.scatter(X_clustering['Area'], X_clustering['Perimeter'], c=df['Best_Cluster'], cmap='viridis', s=50)
    plt


# %%
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score
import joblib



# Use all features for clustering
# X_clustering = df.drop("Class", axis=1)
# true_labels = df['Class']
# print(X_clustering.head(100))


# Implement K-Means clustering
kmeans = KMeans(n_clusters=7, random_state=42)
best_kmeans=kmeans.fit(X_train)

# Analyze the clusters formed
train_labels = kmeans.labels_


silhouette_avg = silhouette_score(X_test, train_labels)
print(f"Silhouette Score: {silhouette_avg}")
joblib.dump(best_kmeans, "kmeansDropped.pkl")

# Visualize the clusters
# plt.scatter(X_clustering['Area'], X_clustering['Perimeter'], c=df['Cluster'], cmap='viridis', s=50)
# plt.title('K-Means Clustering')
# plt.xlabel('Sepal Length (cm)')
# plt.ylabel('Sepal Width (cm)')
# plt.show()




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

        plt.scatter(X_clustering[attribute1], X_clustering[attribute2], c=df['Cluster'], cmap='viridis', s=50)
        plt.title(f'K-Means Clustering - {attribute1} vs {attribute2}')
        plt.xlabel(attributes[i])
        plt.ylabel(attributes[j])
        plt.show()



