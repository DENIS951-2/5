#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from pandas import Series
import pandas_profiling

from datetime import datetime, timedelta

import itertools

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from statsmodels.graphics.mosaicplot import mosaic
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score 
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.feature_selection import RFE
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler




# In[2]:


import warnings

warnings.filterwarnings('ignore')


# ## Мы располагаем следующей информацией из анкетных данных заемщиков:
# 
# **client_id** - идентификатор клиента,  
# **education** - уровень образования,  
# **sex** - пол заемщика,  
# **age** - возраст заемщика,  
# **car** - флаг наличия автомобиля,  
# **car**_type - флаг автомобиля иномарки,  
# **decline_app_cnt** - количество отказанных прошлых заявок,  
# **good_work** - флаг наличия "хорошей" работы,  
# **bki_request_cnt** - количество запросов в БКИ,  
# **home_address** - категоризатор домашнего адреса,  
# **work_address** - категоризатор рабочего адреса,  
# **income**- доход заемщика,  
# **foreign_passport** - наличие загранпаспорта,  
# **sna** - связь заемщика с клиентами банка,  
# **first_time** - давность наличия информации о заемщике,  
# **score_bki** - скоринговый балл по данным из БКИ,  
# **region_rating** - рейтинг региона,  
# **app_date** - дата подачи заявки,  
# **default** - флаг дефолта по кредиту.

# In[3]:


train = pd.read_csv('train.csv')
train.insert(0, '_test', False)

test = pd.read_csv('test.csv')
test.insert(0, '_test', True)

data = pd.concat([train, test], ignore_index=True)
# unique values count, first 10 unique values, null values count, type
data.agg({'nunique', lambda s: s.unique()[:10]})    .append(pd.Series(data.isnull().sum(), name='null'))    .append(pd.Series(data.dtypes, name='dtype'))    .transpose()


# In[4]:


train['education'].value_counts().plot.barh()


# In[5]:


data['education'].value_counts().plot.barh()


# **заменяем пропуски на более часто встречаюшиеся**

# In[6]:


data.education.fillna('SCH', inplace=True)
train.education.fillna('SCH', inplace=True)


# **проверим что у нас все прописалось**

# In[7]:


data.education.unique()


# **делим образование на высшее (1) и школьное (0)**

# In[8]:


data['education'] = data['education'].replace('SCH', 1 )
data['education'] = data['education'].replace('GRD', 0 )
data['education'] = data['education'].replace('UGR', 0 )
data['education'] = data['education'].replace('PGR', 0 )
data['education'] = data['education'].replace('ACD', 0 )
data


# In[9]:


data.education.unique()


# **названия столбцов**

# In[10]:


train.columns


# **Сгруппируем признаки для упрощения обработки.**

# In[11]:


target = 'default'
# бинарные переменные
bin_features = ['sex', 'car', 'car_type', 'good_work', 'foreign_passport']
# категорианальные переменные
cat_features = ['education', 'home_address', 'work_address', 'sna', 'first_time', 'region_rating']
# числовые переменные
num_features = ['age', 'decline_app_cnt', 'score_bki', 'bki_request_cnt', 'income']


# **Посмотрим на распределение количественных признаков.**

# In[12]:


def plot_grid(nplots, max_cols=2, figsize=(800/72, 600/72)):
    ncols = min(nplots, max_cols)
    nrows = (nplots // ncols) + min(1, (nplots % ncols))
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize, constrained_layout=True)
    if nrows == 1:
        return axs
    return [axs[index // ncols, index % ncols] for index in range(nplots)]


# In[13]:


for column, ax in zip(num_features, plot_grid(len(num_features))):
    data[column].plot(kind='hist', ax=ax, title=column)
    ax.set_ylabel(None)


# **Распределение дохода имеет "длинный хвост" и большой масштаб относительно других признаков,  
# поэтому применим функцию нормализации.**

# In[14]:


data.income = data.income.transform(np.log)
data.income.plot(kind='hist');


# In[15]:


sns.heatmap(train[num_features].corr().abs(), vmin=0, vmax=1, annot=True)


# **Наблюдается положительная корреляция целевой переменной с количеством отказов,  
# что может объясняться самим процессом выдачи кредитов. Но в целом численные признаки скоррелированы слабо.  
# Проверим теперь значимость численных признаков.**

# In[16]:


imp_num = Series(f_classif(train[num_features], train['default'])[0], index = num_features)
imp_num.sort_values(inplace = True)
imp_num.plot(kind = 'barh')


# **Обратим внимание на то, что большее значение имеют признаки количества обращений и рейтинг бюро крединтных историй.**

# In[17]:


for feature, ax in zip(cat_features, plot_grid(len(cat_features))):
    mosaic(data[~data._test], [feature, target], ax=ax, title=feature)


# In[18]:


for feature, ax in zip(bin_features, plot_grid(len(bin_features))):
    mosaic(data[~data._test], [feature, target], ax=ax, title=feature)


# In[19]:


from sklearn.preprocessing import LabelEncoder, OrdinalEncoder




label_encoder = LabelEncoder()
for column in bin_features:
    data[column] = label_encoder.fit_transform(data[column])


# In[20]:


mi = mutual_info_classif(data[~data._test][cat_features+bin_features], data[~data._test][target], discrete_features=True)
pd.Series(mi, index=cat_features+bin_features).sort_values(ascending=False).plot(kind='bar')


# In[21]:


def prepare_space(df, num_features, bin_features, cat_features, target):
    X_num = StandardScaler().fit_transform(df[num_features].values)
    X_bin = df[bin_features].values
    X_cat = OneHotEncoder(sparse=False).fit_transform(df[cat_features].values)
    X = np.hstack([X_num, X_bin, X_cat])
    Y = df[target].values
    return X, Y


# In[22]:


X, Y = prepare_space(data[~data._test], num_features, bin_features, cat_features, target)


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2)


# **Воспользуемся логистической регрессией, как относительно простым и быстрым алгоритмом машинного обучения.**

# In[24]:


model = LogisticRegression( solver='liblinear').fit(X_train, y_train)


# **Оценим производительность модели по рабочей характеристике ROC.**

# In[25]:


def plot_roc_curve(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


# In[26]:


proba = model.predict_proba(X_test)[:, 1]
plot_roc_curve(y_test, proba)


# **Производительность нашей модели кажется неплохой,  
# но вернемся в предметную область и взглянем на таблицу сопряженности оценки модели и реальных данных.**

# In[32]:


def plot_cmatrix(estimator, X, y, title=None, **kwargs):
    disp = plot_confusion_matrix(model, X_test, y_test, **kwargs)
    disp.ax_.set_ylabel('Истинный класс')
    disp.ax_.set_xlabel('Предсказанный класс')
    if title:
        disp.ax_.set_title(title)


# In[33]:


axs = plot_grid(2, figsize=(800/72, 300/72))
plot_cmatrix(model, X_test, y_test, title='Абсолютные значения', display_labels=['0', '1'], ax=axs[0]);
plot_cmatrix(model, X_test, y_test, title='Нормализованные значения', normalize='pred', display_labels=['0', '1'], ax=axs[1])


# **Заметна большая доля (13%) ложно-отрицательных значений, что может привести к убыткам из-за вышедших на дефолт кредитов,       выданных потенциально ненадежным клиентам.**
# 
# **Но прежде чем идти на компромисс в выборе определяющего класс порога для векторов вероятностей,  
# попробуем оптимизировать модель путем подбора параметров регрессии.**

# In[34]:


estimator = LogisticRegression(max_iter=200)

param_grid = [
    {
        'penalty': ['l1'],
        'solver': ['liblinear', 'saga'],
        'class_weight':['none', 'balanced'], 
        'multi_class': ['auto','ovr'],
    },
    {
        'penalty': ['l2'],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'class_weight':['none', 'balanced'], 
        'multi_class': ['auto','ovr'],
    },
    {
        'penalty': ['elasticnet'],
        'solver': ['saga'],
        'class_weight':['none', 'balanced'], 
        'multi_class': ['auto','ovr'],
    },
    {
        'penalty': ['none'],
        'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
        'class_weight':['none', 'balanced'], 
        'multi_class': ['auto','ovr'],
    },
]

gs = GridSearchCV(estimator, param_grid, n_jobs=-1, scoring='f1', cv=5).fit(X_train, y_train)


# **Сразу испытаем найденные параметры на использованной ранее выборке.**

# In[35]:


params = gs.best_estimator_.get_params()
model = LogisticRegression(**params).fit(X_train, y_train)


# In[36]:


proba = model.predict_proba(X_test)[:, 1]
plot_roc_curve(y_test, proba)


# In[37]:


axs = plot_grid(2, figsize=(800/72, 300/72))
plot_cmatrix(model, X_test, y_test, title='Абсолютные значения', display_labels=['0', '1'], ax=axs[0]);
plot_cmatrix(model, X_test, y_test, title='Нормализованные значения', normalize='pred', display_labels=['0', '1'], ax=axs[1])


# **Производительность модели не улучшилась, и мы все же пришли к компромиссу:    
# уменьшили долю риска по дефолтным кредитам за счет увеличения ложно-положительных исходов.**
# 
# **Опубликуем результаты.**

# In[38]:


S, _ = prepare_space(data[data._test], num_features, bin_features, cat_features, target)
predict_submission = model.predict_proba(S)[:,1]


# In[39]:


sample_submission = data[data._test][['client_id', target]]
sample_submission[target] = predict_submission
sample_submission.to_csv('submission.csv', index=False)


# In[ ]:




