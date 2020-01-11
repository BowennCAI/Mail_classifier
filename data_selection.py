import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

## Read csvs
train_df = pd.read_csv('train.csv', index_col=0)
test_df = pd.read_csv('test.csv', index_col=0)

#%%

## Handle missing values
train_df.fillna('NA', inplace=True)
test_df.fillna('NA', inplace=True)

#%%

## Filtering column "mail_type"
train_x = train_df[['org','tld','mail_type']]
train_y = train_df[['label']]

test_x = test_df[['org','tld','mail_type']]

## PCA
data_train = np.array(train_df.iloc[:, [3,4,6,7,8,9]])
data_test = np.array(test_df.iloc[:, [3,4,6,7,8,9]])

sc = StandardScaler()
sc.fit(data_train)

X_train_std = sc.transform(data_train)
X_test_std = sc.transform(data_test)

# pca = PCA(n_components=7)
# a = np.vstack((X_train_std, X_test_std))
# pca.fit(a)
#
# X_train_std = a[0:25066]
# X_test_std = a[25066:35811]

feat_enc = OneHotEncoder()
feat_enc.fit(np.vstack((train_x, test_x)))
train_x_featurized = feat_enc.transform(train_x).A
test_x_featurized = feat_enc.transform(test_x).A


X_train = np.hstack((X_train_std, train_x_featurized))
X_test = np.hstack((X_test_std, test_x_featurized))

print(train_df.corr("spearman"))
train_df.corr("spearman").to_csv("num_corelation_label.csv", index=True, index_label='Id')



#########
# RandomForest
# clf_rdf = RandomForestClassifier()
# clf_rdf.fit(X_train, train_y)
#
# pred_y = clf_rdf.predict(X_test)
#
# pred_df = pd.DataFrame(pred_y, columns=['label'])
# pred_df.to_csv("rdf_submission.csv", index=True, index_label='Id')

######

#########
# GradientBoosting
# clf_GB = GradientBoostingClassifier()
# clf_GB.fit(X_train,train_y)
#
# pred_y = clf_GB.predict(X_test)
#
# pred_df = pd.DataFrame(pred_y, columns=['label'])
# pred_df.to_csv("GB_submission.csv", index=True, index_label='Id')
######


#########
# clf = MLPClassifier(activation='tanh', max_iter=500, batch_size=128)
# # clf = MLPClassifier(batch_size=128)
# clf.fit(X_train, train_y)
# pred_y = clf.predict(X_test)
# pred_df = pd.DataFrame(pred_y, columns=['label'])
# pred_df.to_csv("nn_submission.csv", index=True, index_label='Id')

######

#
# print(cross_val_score(clf_rdf, X_train, train_y, cv=3))