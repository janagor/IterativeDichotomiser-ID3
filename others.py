from ucimlrepo import fetch_ucirepo
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from yellowbrick.model_selection import FeatureImportances


mushroom = fetch_ucirepo(id=73) 

X= mushroom.data.features 
y= mushroom.data.targets 

df = pd.concat([X,y], axis =1)

for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
print(df.head())

X_train= df.drop(columns=['poisonous'], axis=1)
y_train= df['poisonous']

model = RandomForestRegressor(random_state=20)
model.fit(X_train, y_train)

importance = pd.Series(
    data = model.feature_importances_,
    index = X_train.columns,
)
importance.sort_values(inplace=True, ascending=False)
importance.plot.bar(figsize=(15,14), fontsize=14)
plt.title("Importance based on Random forest. Mushroom.", fontsize=20)
plt.ylabel("Importance value", fontsize=20)
plt.savefig(f'importances_mush.png')
plt.show()

from sklearn.model_selection import train_test_split

SEED = 42
X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.2, random_state=SEED
)

rfc_ = RandomForestClassifier(
    n_estimators=900, max_depth=7, random_state=SEED
)
rfc_.fit(X_train, y_train)
y_pred = rfc_.predict(X_test)

cm_ = confusion_matrix(y_test, y_pred)
sns.heatmap(cm_, annot=True, fmt='d')
print(classification_report(y_test,y_pred))
plt.show()
###############################################################################
# # fetch dataset 
# breast_cancer = fetch_ucirepo(id=14) 
#   
# # data (as pandas dataframes) 
# X_bre = breast_cancer.data.features 
# y_bre = breast_cancer.data.targets 
#   
# # metadata 
# print(breast_cancer.metadata) 
#   
# # variable information 
# print(breast_cancer.variables) 




mushroom = fetch_ucirepo(id=14) 

X= mushroom.data.features 
y= mushroom.data.targets 

df = pd.concat([X,y], axis =1)

for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
print(df.head())

X_train= df.drop(columns=['Class'], axis=1)
y_train= df['Class']

model = RandomForestRegressor(random_state=20)
model.fit(X_train, y_train)

importance = pd.Series(
    data = model.feature_importances_,
    index = X_train.columns,
)
importance.sort_values(inplace=True, ascending=False)
importance.plot.bar(figsize=(15,14), fontsize=14)
plt.title("Importance based on Random forest. Breast cancer.", fontsize=20)
plt.ylabel("Importance value", fontsize=20)
plt.savefig(f'importances_bre.png')
plt.show()

from sklearn.model_selection import train_test_split

SEED = 42
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, 
                                                    test_size=0.2, 
                                                    random_state=SEED)

rfc_ = RandomForestClassifier(n_estimators=900, 
                             max_depth=7,
                             random_state=SEED)
rfc_.fit(X_train, y_train)
y_pred = rfc_.predict(X_test)

cm_ = confusion_matrix(y_test, y_pred)
sns.heatmap(cm_, annot=True, fmt='d')
print(classification_report(y_test,y_pred))
