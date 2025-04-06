import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import *
import sys
import pylab as pl
data = {
    'Student_ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva', 'Frank', 'Grace', 'Henry', 'Ivy', 'Jack'],
    'Age': [18, 19, 18, 20, 19, 18, 20, 19, 18, 19],
    'Gender': ['F', 'M', 'M', 'M', 'F', 'M', 'F', 'M', 'F', 'M'],
    'Study_Hours': [15, 12, 18, 10, 20, 8, 25, 15, 10, 12],
    'Sleep_Hours': [7, 6, 8, 5, 9, 4, 8, 7, 6, 5],
    'Test_Score': [85, 78, 92, 72, 88, 65, 95, 82, 70, 75],
    'Absences': [2, 5, 1, 3, 0, 8, 1, 2, 4, 6]
}
df = pd.DataFrame(data)
print(df.head())

#Finding any missing values
print(df.isnull().sum())
numerical_cols = ['Age','Study_Hours','Sleep_Hours','Test_Score','Absences']
pl.figure(figsize=(15,5))
pl.suptitle('Before Outlier Handling',y=1.02)
for i, col in enumerate(numerical_cols,1):
    pl.subplot(1,len(numerical_cols),i)
    sns.boxplot(y=df[col])
pl.tight_layout()
#pl.show()
def handle_outliers(df,cols,method='cap'):

    if method == 'cap':
        for col in cols:
            Q1=df[col].quantile(0.25)
            Q3=df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5*IQR
            upper_bound = Q3 + 1.5*IQR
            df[col]=np.where(df[col]<lower_bound,lower_bound,np.where(df[col]>upper_bound,upper_bound,df[col]))
    elif method == 'remove':
        for col in cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5*IQR
            upper_bound = Q1 + 1.5*IQR

            df = df[(df[col]>=lower_bound) &(df[col]<=upper_bound)]

    return df
df_clean = handle_outliers(df,numerical_cols,method='cap')
print(df_clean)
pl.figure(figsize=(15, 5))
pl.suptitle('After Outlier Capping', y=1.02)
for i, col in enumerate(numerical_cols, 1):
    pl.subplot(1, len(numerical_cols), i)
    sns.boxplot(y=df_clean[col])
pl.tight_layout()
#pl.show()

df['Productivity'] = df['Study_Hours']/df['Sleep_Hours']
print(df.head())
#Exploratory Data Analysis
#Correlation Analysis
pl.figure(figsize=(8,6))
sns.heatmap(df[numerical_cols + ['Productivity']].corr(), annot=True, cmap='coolwarm')
pl.title('Correlation Matrix')
#pl.show()

#Distribution of Test score
pl.figure(figsize=(10,5))
sns.histplot(df['Test_Score'],bins = 10,kde=True)
pl.title('Distribution of Test Score')
pl.xlabel('Test Score')
pl.ylabel('Count')
#pl.show()

#Relationship between Study Hours and Test Score
pl.figure(figsize=(10, 5))
sns.scatterplot(x='Study_Hours', y='Test_Score', hue='Gender', data=df, s=100)
pl.title('Study Hours vs Test Score')
pl.xlabel('Study Hours')
pl.ylabel('Test Score')
#pl.show()

#Average Test Score by Gender
pl.figure(figsize=(8, 5))
sns.barplot(x='Gender', y='Test_Score', data=df, errorbar=None)
pl.title('Average Test Score by Gender')
pl.xlabel('Gender')
pl.ylabel('Average Test Score')
#pl.show()
#pairplot for multiple relationships
sns.pairplot(df[['Age','Study_Hours','Sleep_Hours','Test_Score','Gender']],hue='Gender')
#pl.show()

#Does more study time lead to higher scores?
sns.lmplot(x='Study_Hours',y='Test_Score',data=df)
pl.title("Study Hours vs Test Score with Regression Line")
#pl.show()

#How does sleep affect performance
pl.figure(figsize=(10, 5))
sns.scatterplot(x='Sleep_Hours', y='Test_Score', hue='Gender', data=df, s=100)
pl.title('Sleep Hours vs Test Score')
#pl.show()

#Are absences affecting test scores?
pl.figure(figsize=(10, 5))
sns.regplot(x='Absences', y='Test_Score', data=df)
pl.title('Absences vs Test Score')
pl.show()