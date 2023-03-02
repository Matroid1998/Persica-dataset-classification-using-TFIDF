import hazm
import pandas as pd
import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

### read csv and preprocessing
corpus = hazm.PersicaReader(csv_file="persica.csv")
df = pd.DataFrame(corpus.docs())
stop_words = hazm.stopwords_list("stop words.txt")
stop = open(os.path.join("stop words.txt"),encoding="utf8")
stopd=stop.read()
stop.close()
stopwords = hazm.word_tokenize(stopd)
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
def clean_text(text):
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = ' '.join(word for word in text.split() if word not in stopwords) # delete stopwors from text
    text =text.replace("\u200c"," ")
    return text
###split into train and test
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df["category2"]):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]

print(strat_test_set["category2"].value_counts() / len(strat_test_set))
X_train , y_train = strat_train_set["text"],strat_train_set["category2"]
X_test, y_test =strat_test_set["text"],strat_test_set["category2"]
##SVM
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD

svm_classifier = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', SVC(random_state=42)),
               ])
svm_classifier.fit(X_train, y_train)

from sklearn.metrics import classification_report
y_pred = svm_classifier.predict(X_test)
svm_acc=accuracy_score(y_pred,y_test)
print("SVM")
print('accuracy :' ,svm_acc)
print(classification_report(y_test, y_pred,target_names=df['category2'].unique()))
##SVM with SVD
svm_classifier_with_SVD = Pipeline([('vect', CountVectorizer()),
                            ('tfidf', TfidfTransformer()),
                           ("svd",TruncatedSVD(n_components=500)),
                            ('clf', SVC(random_state=42)),
                                  ])
svm_classifier_with_SVD.fit(X_train, y_train)
y_pred = svm_classifier_with_SVD.predict(X_test)
svm_svd_acc=accuracy_score(y_pred, y_test)
print("SVM with SVD")
print('accuracy : ',svm_svd_acc)
print(classification_report(y_test, y_pred,target_names=df['category2'].unique()))
###naive bayes
from sklearn.naive_bayes import MultinomialNB
nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
               ])
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)
naive_bayes_acc=accuracy_score(y_pred,y_test)
print("naive bayes")
print('accuracy : ',naive_bayes_acc)
print(classification_report(y_test, y_pred,target_names=df['category2'].unique()))
###SGD
from sklearn.linear_model import SGDClassifier

sgd = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('SVD',TruncatedSVD(n_components=(500))),
                ('clf', SGDClassifier(random_state=42)),
                ])
sgd.fit(X_train, y_train)

y_pred = sgd.predict(X_test)
SGD_acc=accuracy_score(y_pred,y_test)
print("SGD")
print('accuracy : ', SGD_acc)
print(classification_report(y_test, y_pred,target_names=df['category2'].unique()))
###logistic regression
from sklearn.linear_model import LogisticRegression

logreg = Pipeline([('tfidf', TfidfVectorizer()),
                   ('SVD',TruncatedSVD(n_components=(500))),
                   ('clf', LogisticRegression(random_state=42),),
                   ])
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
Logisticreg_acc=accuracy_score(y_pred,y_test)
print("logistic regression")
print('accuracy : ',Logisticreg_acc )
print(classification_report(y_test, y_pred,target_names=df['category2'].unique()))
###random forest
from sklearn.ensemble import RandomForestClassifier
RF = Pipeline([('tfidf', TfidfVectorizer()),
               ('SVD',TruncatedSVD(n_components=(500))),
                   ('clf', RandomForestClassifier(random_state=42)),
                   ])
RF.fit(X_train, y_train)
y_pred = RF.predict(X_test)
random_forest_acc=accuracy_score(y_pred,y_test)
print("random forest")
print('accuracy : ', random_forest_acc)
print(classification_report(y_test, y_pred,target_names=df['category2'].unique()))
###ensemble
from sklearn.ensemble import VotingClassifier
voting_clf =VotingClassifier(estimators=[('svmc',svm_classifier_with_SVD),('lgd',logreg),('nb',nb)
                                         ],voting='hard')
voting_clf.fit(X_train,y_train)
y_pred=voting_clf.predict(X_test)
print("ensemble")
print('accuracy %s' % accuracy_score(y_pred, y_test))
###accuracy table
plt.figure(figsize=(4, 1), dpi=200)
table = plt.table(cellText= [
    [
        "SVM",
        svm_acc
    ],
    [
        'SVM with svd',
        svm_svd_acc
    ],
    [
        "Naive bayes",
        naive_bayes_acc
    ],
    [
        "Stochastic Gradient Descent",
        SGD_acc
    ],
    [
        'logistic regression',
        Logisticreg_acc
    ],
    [
        "Random Forest",
        random_forest_acc
    ],],colLabels=["Method", "Accuracy"],)
table.set_fontsize(11)
table.scale(2, 2)
plt.axis("off")
plt.show()
