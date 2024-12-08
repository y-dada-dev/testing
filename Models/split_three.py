import pandas as pd
import numpy as np

excel_file_path = 'E:/school/3cs/ml/mini projet/ml2/dataset.xlsx'
df = pd.read_excel(excel_file_path)


fraction_for_dataset1 = 0.9997
df1 = df.sample(frac=fraction_for_dataset1, random_state=42)
df2 = df.drop(df1.index)

df2.to_excel('E:/school/3cs/ml/mini projet/ml2/dataset_test.xlsx', index=False)
