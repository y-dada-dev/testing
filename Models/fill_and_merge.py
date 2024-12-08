import pandas as pd

file_path1 = 'E:/school/3cs/ml/mini projet/ml2/ml/dataset1.xlsx'
file_path2 = 'E:/school/3cs/ml/mini projet/ml2/ml/dataset2.xlsx'



# Read the Excel file into a DataFrame
df1 = pd.read_excel(file_path1)
df2 = pd.read_excel(file_path2)


# Working With dataset2 
new_column_names_dataset1 = {'area': 'Area', 'perimeter': 'Perimeter', 'MajorAL': 'MajorAxisLength', 'MinorAL': 'MinorAxisLength', 'AspectRation': 'AspectRation', 'Eccentricity': 'Eccentricity', 'ConvexArea': 'ConvexArea', 'EquivDiameter': 'EquivDiameter', 'Extent': 'Extent', 'Solidity': 'Solidity', 'Roundness': 'Roundness', 'compactness': 'Compactness', 'shapeFactor1': 'ShapeFactor1', 'shapeFactor2': 'ShapeFactor2', 'shapeFactor3': 'ShapeFactor3', 'shapeFactor4': 'ShapeFactor4', 'Class': 'Class'}


df2 = df2.rename(columns=new_column_names_dataset1)
most_frequent_values = df2.mode().iloc[0]

# Fill null values with the most frequent value in each column
df2_filled = df2.fillna(most_frequent_values)



df = pd.concat([df1, df2_filled], ignore_index=True)
df.to_excel('E:/school/3cs/ml/mini projet/ml2/ml/df.xlsx', index=False)


# Display basic statistics about the dataset

print("\nDataset Statistics:")
print(df.describe())
