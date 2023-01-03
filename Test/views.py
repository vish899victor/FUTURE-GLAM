from django.shortcuts import render, redirect
from .models import answers
from sklearn.tree import DecisionTreeClassifier
from Login import views

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
import warnings

warnings.filterwarnings('ignore')

data = 'ml_model/EDI1.csv'

df = pd.read_csv('ml_model/EDI1.csv',header= None)
# df.shape

# df.head()
# df.info()

df.isnull().sum()
def clean_dataset(data):
    assert isinstance(data, pd.DataFrame), "df needs to be a pd.DataFrame"
    data.dropna(inplace=True)
    indices_to_keep = ~data.isin([np.nan, np.inf, -np.inf]).any(1)
    return data[indices_to_keep].astype(np.float64)

df  = clean_dataset(df)
X = df.iloc[:,0:60]

y = df.iloc[:,60]

# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# check the shape of X_train and X_test

# X_train.shape, X_test.shape


# X_train.dtypes

# X_train.head()

from sklearn.tree import DecisionTreeClassifier
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)


# fit the model
clf_gini.fit(X_train, y_train)

y_pred_gini = clf_gini.predict(X_test)

from sklearn.metrics import accuracy_score

# print('Model accuracy score with criterion gini index: {0:0.4f}'. format(accuracy_score(y_test, y_pred_gini)))

y_pred_train_gini = clf_gini.predict(X_train)

# y_pred_train_gini
# print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_gini)))
# print('Training set score: {:.4f}'.format(clf_gini.score(X_train, y_train)))

# print('Test set score: {:.4f}'.format(clf_gini.score(X_test, y_test)))
clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)


# fit the model
clf_en.fit(X_train, y_train)


# Create your views here.
def test(request):
    if request.GET:
        ans = answers()
        ans.question0 = int(request.GET['question0'])
        ans.question1 = int(request.GET['question1'])
        ans.question2 = int(request.GET['question2'])
        ans.question3 = int(request.GET['question3'])
        ans.question4 = int(request.GET['question4'])
        ans.question5 = int(request.GET['question5'])
        ans.question6 = int(request.GET['question6'])
        ans.question7 = int(request.GET['question7'])
        ans.question8 = int(request.GET['question8'])
        ans.question9 = int(request.GET['question9'])
        ans.question10 = int(request.GET['question10'])
        ans.question11 = int(request.GET['question11'])
        ans.question12 = int(request.GET['question12'])
        ans.question13 = int(request.GET['question13'])
        ans.question14 = int(request.GET['question14'])
        ans.question15 = int(request.GET['question15'])
        ans.question16 = int(request.GET['question16'])
        ans.question17 = int(request.GET['question17'])
        ans.question18 = int(request.GET['question18'])
        ans.question19 = int(request.GET['question19'])
        ans.question20 = int(request.GET['question20'])
        ans.question21 = int(request.GET['question21'])
        ans.question22 = int(request.GET['question22'])
        ans.question23 = int(request.GET['question23'])
        ans.question24 = int(request.GET['question24'])
        ans.question25 = int(request.GET['question25'])
        ans.question26 = int(request.GET['question26'])
        ans.question27 = int(request.GET['question27'])
        ans.question28 = int(request.GET['question28'])
        ans.question29 = int(request.GET['question29'])
        ans.question30 = int(request.GET['question30'])
        ans.question31 = int(request.GET['question31'])
        ans.question32 = int(request.GET['question32'])
        ans.question33 = int(request.GET['question33'])
        ans.question34 = int(request.GET['question34'])
        ans.question35 = int(request.GET['question35'])
        ans.question36 = int(request.GET['question36'])
        ans.question37 = int(request.GET['question37'])
        ans.question38 = int(request.GET['question38'])
        ans.question39 = int(request.GET['question39'])
        ans.question40 = int(request.GET['question40'])
        ans.question41 = int(request.GET['question41'])
        ans.question42 = int(request.GET['question42'])
        ans.question43 = int(request.GET['question43'])
        ans.question44 = int(request.GET['question44'])
        ans.question45 = int(request.GET['question45'])
        ans.question46 = int(request.GET['question46'])
        ans.question47 = int(request.GET['question47'])
        ans.question48 = int(request.GET['question48'])
        ans.question49 = int(request.GET['question49'])
        ans.question50 = int(request.GET['question50'])
        ans.question51 = int(request.GET['question51'])
        ans.question52 = int(request.GET['question52'])
        ans.question53 = int(request.GET['question53'])
        ans.question54 = int(request.GET['question54'])
        ans.question55 = int(request.GET['question55'])
        ans.question56 = int(request.GET['question56'])
        ans.question57 = int(request.GET['question57'])
        ans.question58 = int(request.GET['question58'])
        ans.question59 = int(request.GET['question59'])

        cardict = {101:'HEALTHCARE', 102:'FINANCE',103:'BUSSINESS',104:'TECHNOLOGY',105:'MULTIMEDIA',106:'LEGAL',107:'PUBLIC SERVICE',108:'ARTS'
            ,109:'CULINARY',110:'EDUCATION',111:'COMMUNICATION',112:'SOCIAL SCIENCE',113:'SCIENCE',114:'ENGINEERING',115:'TRADE VOCATION'}
        answer=[ans.question1,ans.question0,ans.question2,ans.question3,ans.question4,ans.question5,ans.question6,ans.question7,ans.question8,ans.question9,ans.question10,ans.question11,
                ans.question12,ans.question13,ans.question14,ans.question15,ans.question16,ans.question17,ans.question18,ans.question19,ans.question20,ans.question21,ans.question22,ans.question23,ans.question24,ans.question25,
                ans.question26,ans.question27,ans.question28,ans.question29,ans.question30,ans.question31,ans.question32,ans.question33,ans.question34,ans.question35,
                ans.question36,ans.question37,ans.question38,ans.question39,ans.question40,ans.question41,ans.question42,ans.question43,ans.question44,
                ans.question45,ans.question46,ans.question47,ans.question48,ans.question49,ans.question50,ans.question51,ans.question52,ans.question53,ans.question54,
                ans.question55,ans.question56,ans.question57,ans.question58,ans.question59]
        
        y_pred_en = clf_en.predict([answer])
        y = cardict[y_pred_en[0]]
        
        return render(request, 'admissions1.html',{'answer':y}) 
    else:
        return render(request, 'admissions.html')
def login(request):
        return redirect('login')


    