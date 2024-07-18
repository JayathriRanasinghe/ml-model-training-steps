import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


df= pd.read_csv('train.csv')

# number of rows and number of columns, so samples and features
df.shape

# visualizing first rows of dataset
df.head()

df.describe()

#column names, number of non-null values, Dtype and memory usage
df.info() 

# columns and null value count in each column
df.isnull().sum() 

#unique categories in this feature
df['Sex'].unique()
df['Sex'].value_counts() #number of duplications for each unique category

#selecting columns considering the dtype
Numerical_cols = df.select_dtypes(include=['float64', 'int64'])


#handling missing entries
df['Age'].fillna(df['Age'].mean(), inplace=True) #replacing with mean (for sequential)
df['Fare'].fillna(df['Fare'].median(), inplace=True) #replacing with median (for sequential)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True) #replacing with mode (for categorical features)
df['Cabin'].fillna("NO", inplace=True) #replacing with different label

#dropping features
df = df.drop(columns=['Name','Ticket','Cabin', 'Embarked'])

#one hot encoding
encoder = OneHotEncoder(sparse=False)
encoded_data = encoder.fit_transform(df[['Sex', 'Embarked']])
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['Sex', 'Embarked']))
df_final = pd.concat([df.drop(columns=['Sex', 'Embarked']), encoded_df], axis=1)

#Label encoding
encoder = LabelEncoder()
df['Embarked_Encoded'] = encoder.fit_transform(df['Embarked'])
df.drop(columns=['Embarked'])

#assigning some labels manually
df['Sex'] = df['Sex'].replace({'male': 0, 'female': 1})
