import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("D:/Users/sefa.erkan/Desktop/Credit_Card/credit_card_transactions.csv")
df.head(5)
df.info()

df.drop(columns=['Unnamed: 0', 'trans_date_trans_time','cc_num',
                 'first','last','merchant','zip','city_pop',
                 'trans_num','unix_time','merch_zipcode','dob',
                 'lat','long','merch_lat','merch_long'],inplace=True)

df.describe()
# Tekrarlanan satırların sayısı
df.duplicated().sum()

df.head()

df.describe(include='all')

# Analiz

plt.figure(figsize=[10,8])
sns.countplot(data=df, x='category')
plt.xticks(rotation=45)
plt.xlabel('Categories')
plt.ylabel('Frequency')
plt.title('Categories Distribution')
plt.show()

sns.countplot(data=df, x='gender')
plt.title('Gender Distribution')
plt.show()

sns.countplot(data=df, x='is_fraud')
plt.show()

# Preprocessing
encoder = LabelEncoder()

df.category = encoder.fit_transform(df.category)
df.gender = encoder.fit_transform(df.gender)
df.street = encoder.fit_transform(df.street)
df.city = encoder.fit_transform(df.city)
df.state = encoder.fit_transform(df.state)
df.job = encoder.fit_transform(df.job)

X = df.drop(columns=['is_fraud'],axis=1)
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, shuffle=True, random_state=42)

## 1.Logistic Regression
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

model.score(X_train, y_train)

model.score(X_test,y_test)

y_pred = model.predict(X_test)

accuracy_score(y_pred, y_test)

print(classification_report(y_pred,y_test))

cm = confusion_matrix(y_pred, y_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()

## 2. Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

model.score(X_train,y_train)

model.score(X_test, y_test)

y_pred = model.predict(X_test)

accuracy_score(y_pred, y_test)

print(classification_report(y_pred,y_test))

cm = confusion_matrix(y_pred, y_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()




