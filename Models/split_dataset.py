import pandas as pd
import numpy as np
# Load the Excel file into a DataFrame
excel_file_path = 'E:/school/3cs/ml/mini projet/ml2/ml/dataset.xlsx'
df = pd.read_excel(excel_file_path)

# Split the DataFrame into two datasets (e.g., 80% and 20%)
# You can adjust the fraction to suit your needs
fraction_for_dataset1 = 0.6
df1 = df.sample(frac=fraction_for_dataset1, random_state=42)
df2 = df.drop(df1.index)

# Add some missing values (NaN) to the DataFrame
# You can adjust the fraction to control the percentage of missing values
#fraction_missing_values_df2 = 0.1
#mask_df2 = np.random.rand(*df2.shape) < fraction_missing_values_df2
#df2[mask_df2] = np.nan
#df2[mask_df2] = "null"

# Rename columns for each dataset as needed
# Replace 'new_column_name' with the desired column names
new_column_names_dataset2 = {'Area': 'area', 'Perimeter': 'perimeter', 'MajorAxisLength': 'MajorAL', 'MinorAxisLength': 'MinorAL', 'AspectRation': 'AspectRation', 'Eccentricity': 'Eccentricity', 'ConvexArea': 'ConvexArea', 'EquivDiameter': 'EquivDiameter', 'Extent': 'Extent', 'Solidity': 'Solidity', 'Roundness': 'Roundness', 'Compactness': 'compactness', 'ShapeFactor1': 'shapeFactor1', 'ShapeFactor2': 'shapeFactor2', 'ShapeFactor3': 'shapeFactor3', 'ShapeFactor4': 'shapeFactor4', 'Class': 'Class'}


df2 = df2.rename(columns=new_column_names_dataset2)

columns_with_nulls = ['area', 'perimeter', 'MajorAL', 'MinorAL', 'AspectRation', 
                      'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent', 'Solidity', 
                      'roundness', 'compactness', 'shapeFactor1', 'shapeFactor2', 
                      'shapeFactor3', 'shapeFactor4']

# Introduce null values to the specified columns
# Specify the percentage of cells in each column to be set to NaN
percentage_cells_with_nulls = 10

for column in columns_with_nulls:
   # Get the number of cells to set to NaN based on the percentage
    num_cells = int(df2[column].count() * (percentage_cells_with_nulls / 100))
    
    # Get random indices to set to NaN
    random_indices = np.random.choice(df2[column].index, size=num_cells, replace=False)
    
    # Replace the selected cells with NaN
    df2.loc[random_indices, column] = np.nan




# Save the two datasets to separate Excel files
df1.to_excel('E:/school/3cs/ml/mini projet/ml2/ml/clean.xlsx', index=False)
df2.to_excel('E:/school/3cs/ml/mini projet/ml2/ml/trash.xlsx', index=False)
