# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 22:42:56 2026

@author: Beg
"""
#import packeges

from ydata_profiling import ProfileReport
import pandas as pd 
import matplotlib.pyplot as plt  
import numpy as np 
import seaborn as sns
import ast 

#loading data

data_dictionary = pd.read_csv("C:\\Users\\Beg\\Desktop\\proiecti\\Cafe Rewards Offers\\data_dictionary.csv") 

customers = pd.read_csv("C:\\Users\\Beg\\Desktop\\proiecti\\Cafe Rewards Offers\\customers.csv")

events = pd.read_csv("C:\\Users\\Beg\\Desktop\\proiecti\\Cafe Rewards Offers\\events.csv") 

df_new = events["value"].apply(pd.Series) 

events['value'] = events['value'].apply(ast.literal_eval)

value_df = pd.json_normalize(events['value'])

print(value_df.head())  

print(value_df.value_counts())

events = pd.concat([events.drop(columns=['value']), value_df], axis=1)
#value_df = events['value'].apply(pd.Series)
#print(value_df)

offers = pd.read_csv("C:\\Users\\Beg\\Desktop\\proiecti\\Cafe Rewards Offers\\offers.csv")  

print(offers["offer_id"].size)

#clean data and describe(EDA)

print(customers.head())

customers['became_member_on'] = pd.to_datetime(customers['became_member_on'], format='%Y%m%d')

profile = ProfileReport(customers, title="EDA Report")
profile.to_file("C:\\Users\\Beg\\Desktop\\proiecti\\report_cafe_customers.html")

profile_events = ProfileReport(events,title = "EDA events") 
profile_events.to_file("C:\\Users\\Beg\\Desktop\\proiecti\\report_cafe_events.html")

#pip install sweetviz 
#import sweetviz as sv

#report = sv.analyze(customers)
#report.show_html("C:\\Users\\Beg\\Desktop\\proiecti\\report_cafe_customerss.html")


corr = customers[['age','income']].corr()
plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation between Age and Income')
plt.show()


#Data preparation (merging) 

customers = customers[['customer_id', 'age', 'income', 'gender']].fillna(0) 

offers = offers[['offer_id', 'offer_type', 'difficulty', 'reward', 'duration']]
 
completed = events[events['event'] == 'offer completed']
completed = completed[['customer_id', 'offer_id']]
completed['offer_completed'] = 1

# All offers received by the customer
all_offers = events[events['event']=='offer received'][['customer_id','offer id']]
all_offers = all_offers.rename(columns = {"offer id":"offer_id"})

# Add target offer_completed
completed = events[events['event']=='offer completed'][['customer_id','offer_id']]

completed['offer_completed'] = 1

# Combine with the offers received
all_offers = all_offers.merge(completed, on=['customer_id','offer_id'], how='left')
all_offers['offer_completed'] = all_offers['offer_completed'].fillna(0)


df = all_offers.merge(customers, on='customer_id', how='left')

#  gender 

df['gender'] = df['gender'].map({'M':0,'F':1,'O':2})

df = df.merge(offers, on='offer_id', how='left')

df = pd.get_dummies(df, columns=['offer_type'], drop_first=True)

X = df.drop(columns=['customer_id','offer_id','offer_completed'])
y = df['offer_completed']

#split data for train and test parts

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#construction model Random Forest 

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Accuracy
from sklearn.metrics import accuracy_score


print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))


# plot model

importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values(ascending=True).tail(10).plot(kind='barh')
plt.title('Top 10 Feature Importances')
plt.show()



















