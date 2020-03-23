# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# import warnings filter

from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import pickle
import sys

from src.models import LogRegtrain,NBtrain,SGDtrain
from src.cat import catClass
from src.ban import bannedTweets

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("C:/Users/vicky/Downloads/Major Project"))

# Any results you write to the current directory are saved as output.

## Reading the dataset

train = pd.read_csv("D:/Major/data/train_E6oV3lV.csv")
test = pd.read_csv("D:/Major/data/test_tweets_anuFYb8.csv")

train.head()

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

test.head()

tempdf = train

train['label'] = train['label'].astype('category')

train.info()

## Processing the Tweets

# Remove the special characters, numbers etc. (keep only alphabets)
# lemmatize the text


from nltk.stem import WordNetLemmatizer
from nltk import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import re

import nltk

train['text_lem'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]',' ',text)) for text in lis]) for lis in train['tweet']]
test['text_lem'] = [''.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]',' ',text)) for text in lis]) for lis in test['tweet']]

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(train['text_lem'],train['label'])

y_test

vect = TfidfVectorizer(ngram_range = (1,4)).fit(X_train)

vect

vect_transformed_X_train = vect.transform(X_train)
vect_transformed_X_test = vect.transform(X_test)

vect_transformed_X_train
vect_transformed_X_test

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
#from sklearn.metrics import classification_report,confusion_matrix
from time import time

### F1 score is used as an evaluation measure as, when the data is skewed like in this case, where the number of hate speech tweets are very less, accuracy cannot be relied upon.

modelLR = LogisticRegression(C=100,solver='liblinear',max_iter=1000).fit(vect_transformed_X_train,y_train)

predictionsLR = modelLR.predict(vect_transformed_X_test)
sum(predictionsLR==1),len(y_test),f1_score(y_test,predictionsLR)

modelNB = MultinomialNB(alpha=1.7).fit(vect_transformed_X_train,y_train)

predictionsNB = modelNB.predict(vect_transformed_X_test)
sum(predictionsNB==1),len(y_test),f1_score(y_test,predictionsNB)

modelSGD = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3).fit(vect_transformed_X_train,y_train)

predictionsSGD = modelSGD.predict(vect_transformed_X_test)
sum(predictionsSGD==1),len(y_test),f1_score(y_test,predictionsSGD)

#Based on all the above models trained we conclude that the logistic regression(C=100) is the better model amoung them, ergo, we use this model as our final model.

vect = TfidfVectorizer(ngram_range = (1,4)).fit(train['text_lem'])
vect_transformed_train = vect.transform(train['text_lem'])
vect_transformed_test = vect.transform(test['text_lem'])

vect_transformed_test

from sklearn.externals import joblib
import psutil



t0 = time()

#psutilpercent = psutil.virtual_memory()
#print ("\n --> Memory Check Percent:", str(psutilpercent.percent) + "%\n")

#modelLog = LogisticRegression(C=100,solver='liblinear',max_iter=1000).fit(vect_transformed_train,train['label'])
#joblib.dump(modelLog,'savedmodelLog.pkl')

per1,tm1 = LogRegtrain(vect_transformed_train,train)

psutilpercent1 = per1
#psutilpercent = psutil.virtual_memory()

print ("\n --> Memory Check Percent:", str(psutilpercent1.percent) + "%\n")

tm1 = round(time()-t0,3)

#print("Train time: ",round(time()-t0,3),"seconds")
print("Train time for Logistic Regression Classifier: ",tm1,"seconds")

t1 = time()

savedmodelLog = joblib.load('D:/Major/saved_models/savedmodelLog.pkl')
predictionsLog = savedmodelLog.predict(vect_transformed_test)

print("Predict time: ",round(time()-t1, 3),"seconds")

print(predictionsLog)

print('No. of Bytes : ',sys.getsizeof(savedmodelLog))




t0 = time()

#psutilpercent = psutil.virtual_memory()
#print ("\n --> Memory Check Percent:", str(psutilpercent.percent) + "%\n")

#modelNB = MultinomialNB(alpha=1.7).fit(vect_transformed_train,train['label'])
#joblib.dump(modelNB,'savedmodelNB.pkl')

per2,tm2 = NBtrain(vect_transformed_train,train)

psutilpercent2 = per2
#psutilpercent = psutil.virtual_memory()

print ("\n --> Memory Check Percent:", str(psutilpercent2.percent) + "%\n")

#tm2 = round(time()-t0,3)

#print("Train time: ",round(time()-t0,3),"seconds")
print("Train time for Multinomial Naive Bayes Classifier: ",tm2,"seconds")

t1 = time()

savedmodelNB = joblib.load('D:/Major/saved_models/savedmodelNB.pkl')
predictionsNB = savedmodelNB.predict(vect_transformed_test)

print("Predict time: ",round(time()-t1, 3),"seconds")

print('No. of Bytes : ',sys.getsizeof(savedmodelNB))





t0 = time()

#psutilpercent = psutil.virtual_memory()
#print("\n --> Memory Check Percent:", str(psutilpercent.percent) + "%\n")

#modelSGD = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3).fit(vect_transformed_train,train['label'])
#joblib.dump(modelSGD,'savedmodelSGD.pkl')

per3,tm3 = SGDtrain(vect_transformed_train,train)

psutilpercent3 = per3
#psutilpercent = psutil.virtual_memory()

print ("\n --> Memory Check Percent:", str(psutilpercent3.percent) + "%\n")

#tm3 = round(time()-t0,3)

#print("Train time: ",round(time()-t0,3),"seconds")
print("Train time for SGD Classifier: ",tm3,"seconds")

t1 = time()

savedmodelSGD = joblib.load('D:/Major/saved_models/savedmodelSGD.pkl')
predictionsSGD = savedmodelSGD.predict(vect_transformed_test)


print("Predict time: ",round(time()-t1, 3),"seconds")

print('No. of Bytes : ',sys.getsizeof(savedmodelSGD))

submission1 = pd.DataFrame({'id':test['id'],'label':predictionsLog,'Tweet':test['tweet']})
submission2 = pd.DataFrame({'id':test['id'],'label':predictionsNB,'Tweet':test['tweet']})
submission3 = pd.DataFrame({'id':test['id'],'label':predictionsSGD,'Tweet':test['tweet']})

file_name1 = 'D:/Major/outputs/test_predictions1.csv'
file_name2 = 'D:/Major/outputs/test_predictions2.csv'
file_name3 = 'D:/Major/outputs/test_predictions3.csv'

submission1.to_csv(file_name1,index=False)
submission2.to_csv(file_name2,index=False)
submission3.to_csv(file_name3,index=False)

df1 = pd.read_csv('D:/Major/outputs/test_predictions1.csv')
df2 = pd.read_csv('D:/Major/outputs/test_predictions2.csv')
df3 = pd.read_csv('D:/Major/outputs/test_predictions3.csv')

#len(df1)
print('Predictions by Logistic Regression Model: \n')
print(df1.head(60))
#print(df3.head(60))

print('Predictions by Multinomial Naive Bayes Model: \n')
print(df2.head(60))

print('Predictions by SGD Classifier Model: \n')
print(df3.head(60))

from sklearn.metrics import accuracy_score,jaccard_score

print('\n Accuracy of Logistic Regression Model: ',f1_score(tempdf['label'][0:17197],predictionsLog))

print('\n Accuracy of Multinomial Naive Bayes Model: ',f1_score(tempdf['label'][0:17197],predictionsNB))

print('\n Accuracy of SGD Classifier Model: ',f1_score(tempdf['label'][0:17197],predictionsSGD))




hate = df1[df1['label']==1]

non_hate = df1[df1['label']==0]

hate.to_csv('D:/Major/outputs/hate_tweets_pre.csv',index=False)

non_hate.to_csv('D:/Major/outputs/non_hate_tweets.csv',index=False)

hate2 = pd.read_csv('D:/Major/data/hate_tweets_modified.csv')

catDict = catClass(hate2)

print("\nCategory Dictionary: \n")
print('Bad IDs: \n',catDict['Bad'])
print('Worst IDs: \n',catDict['Worst'])
print('IDs to be banned: \n',catDict['Extreme'])


print('\n Number of Bad IDs: ',len(catDict['Bad']))
print('\n Number of Worst IDs: ',len(catDict['Worst']))
print('\n Number of IDs to be banned: ',len(catDict['Extreme']))

ban_list = catDict['Extreme']

bannedTweets(hate2,ban_list)

print('\nIDs to be banned are:\n')
print(ban_list)
