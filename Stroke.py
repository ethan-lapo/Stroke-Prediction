import csv
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('C:/Users/Ethan Lapaczonek/Downloads/healthcare-dataset-stroke-data.csv')
print(df)

print(df.isna())
print(df.describe())
print(df[df["bmi"].isnull()])

#Look at some visuals of BMI to see if there are any outliers 

plt.scatter(df['bmi'])
plt.show()