import numpy as np
import pandas as p
from sklearn import svm, metrics, preprocessing
from sklearn.model_selection import train_test_split, cross_val_score

data = p.read_csv(filepath_or_buffer='train.tsv', sep='\t')
filtered_data = data[['age', 'race', 'sex', 'Y']]

pre_X = filtered_data[['age', 'race', 'sex']]

# データ整形 race編
pre_X.race = pre_X.race.replace('White',1)
pre_X.race = pre_X.race.replace('Black',2)
pre_X.race = pre_X.race.replace('Asian-Pac-Islander',3)
pre_X.race = pre_X.race.replace('Amer-Indian-Eskimo',4)
pre_X.race = pre_X.race.replace('Other',5)

# データ整形 sex編
pre_X.sex = pre_X.sex.replace('Male',1)
pre_X.sex = pre_X.sex.replace('Female',2)

sc = preprocessing.StandardScaler()
sc.fit(pre_X)
X = sc.transform(pre_X)
Y = filtered_data['Y']

svm_result = svm.LinearSVC(loss='hinge', C=1.0, class_weight='balanced')
svm_result.fit(X, Y)

scores = cross_val_score(svm_result, X, Y, cv=10)
print("平均正解率 = ", scores.mean())
print("正解率の標準偏差 = ", scores.std())

X_train, X_test, train_label, test_label = train_test_split(X, Y, test_size=0.1, random_state=0)
svm_result.fit(X_train, train_label)
pre = svm_result.predict(X_test)

ac_score = metrics.accuracy_score(test_label,pre)
print("正答率 = ",ac_score)
