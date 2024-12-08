import pandas as pd

# Load the Excel file into a DataFrame
excel_file_path = 'E:/school/3cs/ml/mini projet/ml2/ml/dataset2.xlsx'

df = pd.read_excel(excel_file_path)

# Find the most frequent value in each column
most_frequent_values = df.mode().iloc[0]

# Fill null values with the most frequent value in each column
df_filled = df.fillna(most_frequent_values)


# Save the filled DataFrame to a new Excel file
df_filled.to_excel('E:/school/3cs/ml/mini projet/ml2/ml/dataset_filled.xlsx', index=False)
