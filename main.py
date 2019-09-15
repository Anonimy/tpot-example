from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np

data = pd.read_csv('bank.csv', sep=';')
data.rename(columns={'y': 'class'}, inplace=True)

data['marital'] = data['marital'].map({'married':0,'single':1,'divorced':2,'unknown':3})
data['default'] = data['default'].map({'no':0,'yes':1,'unknown':2})
data['housing'] = data['housing'].map({'no':0,'yes':1,'unknown':2})
data['loan'] = data['loan'].map({'no':0,'yes':1,'unknown':2})
data['contact'] = data['contact'].map({'telephone':0,'cellular':1})
data['poutcome'] = data['poutcome'].map({'nonexistent':0,'failure':1,'success':2})
data['class'] = data['class'].map({'no':0,'yes':1})

data = data.fillna(-999)

mlb = MultiLabelBinarizer()

job_Trans = mlb.fit_transform([{str(val)} for val in data['job'].values])
education_Trans = mlb.fit_transform([{str(val)} for val in data['education'].values])
month_Trans = mlb.fit_transform([{str(val)} for val in data['month'].values])

data_new = data.drop(['marital','default','housing','loan','contact','poutcome','class','job','education','month'], axis=1)
data_new = np.hstack((data_new.values, job_Trans, education_Trans, month_Trans))

data_class = data['class'].values

training_indices, validation_indices = training_indices, testing_indices = train_test_split(
  data.index,
  stratify=data_class,
  train_size=0.75,
  test_size=0.25
)

tpot = TPOTClassifier(
  population_size=15,
  max_eval_time_mins=0.04,
  max_time_mins=2,
  verbosity=3,
  n_jobs=-1
)
tpot.fit(data_new[training_indices], data_class[training_indices])

score = tpot.score(data_new[validation_indices], data.loc[validation_indices, 'class'].values)
print(score)
