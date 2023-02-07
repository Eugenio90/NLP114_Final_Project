import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from pandas import json_normalize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics

## SETTING PATH VARIABLES
location = r"C:/cs/ds/data2/archive"
database = "relevant_articles.json"

## IMPORTING DATA
os.chdir(location)
with open(database, "r") as f:
    data = json.load(f)
    db = json_normalize(data)



## DATA PRE PROCESSING ##

#Dropping unecesary Variables
db = db.drop(['_id', 'url', 'type', 'lang', 'sum', 'refs', 'pub.$date', 'ret.$date', 'auth', 'text', 'body'], axis=1)

# Changing variable types
db['title'] = db['title'].astype('str')
db['Newspaper'] = db['feed'].astype('str')
db = db.drop(['feed'], axis=1)

# Creating numerical targets
le = LabelEncoder()
db['Num_Newspapers'] = le.fit_transform(db['Newspaper'])
db = db.sort_values('Num_Newspapers')

# Referencing numerical targets with each newspaper, this will be used to for confusion matrix.
reference_newspaper_nums = db['Num_Newspapers'].unique()
reference_newspaper_names = db['Newspaper'].unique()

# Dropping the newspaper variable from the dataframe
db = db.drop(['Newspaper'], axis=1)

# Creating a processed_data variables to store the data for each
Processed_Data = db

### DATA PROCESSING COMPLETE ###


# Vectorizer and Model
vec = CountVectorizer()

# Creating X and Y
x = vec.fit_transform(db['title'])
y = db['Num_Newspapers']


def run_svm(Processed_Data):
    print("<---------- C- Classifier -------------> ")
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(Processed_Data['title'])
    y = Processed_Data['Num_Newspapers']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=0)
    clf = SVC(gamma='auto')
    clf.fit(X_train, y_train)
    yhat = clf.predict(X_test)
    print(metrics.classification_report(y_test, yhat, zero_division=0,
                                        target_names=['AP', 'Atlantic', 'CBC', 'CBS', 'Fox', 'HuffingtonUK',
                                                      'HuffingtonUS', 'LATimes',
                                                      'Skynews', 'WP', 'bbc', 'guardian', 'telegraph']))
    print("Testing for Overfitting")
    yh2 = clf.predict(X_train)
    print("Accuracy Score for predictions on training set")
    print(metrics.accuracy_score(y_train, yh2))


def Random_Forest_Model(x, y , estimator_list):
    print("<---------- Random Forest Classifier -------------> ")
    for each_estimator in estimator_list:
        model = RandomForestClassifier(n_estimators=each_estimator)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10)

        pred_model = model.fit(X_train, y_train)

        yh = pred_model.predict(X_test)

        print(" ")
        print("Random Forest Classifer with estimators = ", each_estimator)
        print(metrics.classification_report(y_test, yh, zero_division=0,
                                            target_names=['AP', 'Atlantic', 'CBC', 'CBS', 'Fox', 'HuffingtonUK',
                                                          'HuffingtonUS', 'LATimes',
                                                          'Skynews', 'WP', 'bbc', 'guardian', 'telegraph']))
        print("Testing for Overfitting")
        yh2 = model.predict(X_train)
        print("Accuracy Score for predictions on training set")
        print(metrics.accuracy_score(y_train, yh2))



def run_naive_Bayes(naive_bayes_hyperparameters, Processed_Data):
    print("<---------- Naive Bayes Multinomial -------------> ")
    y = Processed_Data['Num_Newspapers']
    vectorizer = TfidfVectorizer()
    x = Processed_Data['title']
    x = vectorizer.fit_transform(x)
    for each_hyper in naive_bayes_hyperparameters:

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=0)

        clf = MultinomialNB(alpha=each_hyper)
        clf.fit(X_train, y_train)
        yhat = clf.predict(X_test)
        print(" ")
        print("Naive Bayes Multinomial Classifier with alpha = ", each_hyper)
        print(metrics.classification_report(y_test, yhat, zero_division=0,
                                            target_names=['AP', 'Atlantic', 'CBC', 'CBS', 'Fox', 'HuffingtonUK',
                                                          'HuffingtonUS', 'LATimes',
                                                          'Skynews', 'WP', 'bbc', 'guardian', 'telegraph']))

        print("")
        print("Testing for Overfitting")
        yh2 = clf.predict(X_train)
        print("Accuracy Score for predictions on training set")
        print(metrics.accuracy_score(y_train, yh2))



def run_knn(x, y, Knn_list):
    print("<---------- KNeighborsClassifier -------------> ")
    for each_num in Knn_list:
        knc = KNeighborsClassifier(n_neighbors=10)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10)
        knc.fit(X_train, y_train)
        yh = knc.predict(X_test)
        print("")
        print("KNN with ", each_num, " neighbors")
        print(metrics.classification_report(y_test, yh, zero_division=0,
                                            target_names=['AP', 'Atlantic', 'CBC', 'CBS', 'Fox', 'HuffingtonUK',
                                                          'HuffingtonUS', 'LATimes',
                                                          'Skynews', 'WP', 'bbc', 'guardian', 'telegraph']))
        print("Testing for Overfitting")
        yh2 = knc.predict(X_train)
        print("Accuracy Score for predictions on training set")
        print(metrics.accuracy_score(y_train, yh2))



def run():

    # # Random Forest
    estimator_list = [50, 100, 150]
    Random_Forest_Model(x, y, estimator_list)

    # KNN
    Knn_list = [3, 5, 10]
    run_knn(x, y, Knn_list)

    # Naive Bayes
    naive_bayes_hyperparameters = [0.1, 0.5, 1]
    run_naive_Bayes(naive_bayes_hyperparameters, Processed_Data)

    # SVM
    run_svm(Processed_Data)

run()