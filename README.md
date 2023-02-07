
<h1> Classifying News Titles by Newspaper  </h1>
<h2> Eugenio Sanchez | COSI 114a | Professor Lignos</h2>


### Introduction 

>This paper is the presentation of the results of the final project for the <strong>COSI 114a Natural Language Processing</strong> class with Professor Lignos at Brandeis University during the Fall of 2022. The goal was to apply the techniques learned during the semester. For this, a data set of newspaper articles from Kaggle was chosen, and a set of classification algorithms were selected to classify article titles by newspaper. 

### Data

>The database is named 'News Articles' and was found on Kaggle. It contains 60,795 articles that contain the following features: _id, url, title, feed, type, language, summary, body, text, publication date, release date, and author. For this research, we will omit all features but the title and feed. The title represents the title of the article represented in the newspaper and the feed represents the newspaper that emitted this article.


>The title feature is a string containing words. For 60,795 articles, there is a total of 631,901 words, with each title having a mean of 10 words with a standard deviation of 3.3. The feed represents a total of 13 unique newspapers: 'AP', 'Atlantic', 'CBC', 'CBS', 'Fox', 'HuffingtonUK', 'HuffingtonUS', 'LATimes', 'Skynews', 'WP', 'bbc', 'guardian', and 'telegraph'. 

>An important characteristic  is that the data is unbalanced. In figure 1, you can appreciate the different amount of instances for each class. 

#### Figure 1

| Class / Feed | Count  |
|--------------|--------| 
| HuffingtonUS | 15815  |
| guardian     | 9968   |
| WP           | 8229   |
| telegraph    | 4482   |
| Fox          | 4433   |
| bbc          | 4104   |
| CBS          | 4079   |
| HuffingtonUK | 3702   |
| AP           | 3489   |
| Skynews      | 1341   |
| LATimes      | 687    |
| CBC          | 327    |
| Atlantic     | 139    |


### Models

>For this multiclassification task, the following models where used: Random Forest Classifier, K-Neighbors Classifier, Naive Bays Multinomial Classifier, and C-Support Vector Classification. 

### Data Pre-processing 

>For data pre-processing, unnecessary columns were deleted, keeping only 'title' and 'feed' which were revised for missing values. These two were the only relevant features from the data set for our classification task. For the 'feed' feature, this was transformed using a LabelEconder() function from sklearn preprocessing to make this a numerical categorical variable. The 'title' feature, was handled different depending on the algorithm. For the case of C-Support Vector Classification, the TfidVectorizer() was used and for the rest of the algorithms, the Count_Vectorizer() was used to transform the String variable into a matrix. 

## Results

### Figure 2



| Classifer Algorithm                | Hyperparameter     | Training Accuracy  | Test Accuracy | Test Macro F-1 Score |
|------------------------------------|--------------------|--------------------|---------------|-------------------------|
| Random Forest Classifer            | n_estimators = 50  | 0.99               | 0.45          | 0.29                    |
| Random Forest Classifer            | n_estimators = 100 | 0.99               | 0.46          | 0.28                   |
| Random Forest Classifer            | n_estimators = 150 | 0.99               | 0.48          | 0.31                    |
| K-Neighbors Classifier             | n_neighbors = 3    | 0.31               | 0.29          | 0.12                    |
| K-Neighbors Classifier             | n_neighbors = 5    | 0.32               | 0.29          | 0.11                    |
| K-Neighbors Classifier             | n_neighbors = 10   | 0.32               | 0.29          | 0.11                    |
| Naive Bayes Multinomial Classifier | Alpha = 0.1        | 0.70               | 0.48          | 0.27                    |
| Naive Bayss Multinomial Classifier | Alpha = 0.5        | 0.56               | 0.46          | 0.20                    |
| Naive Bayes Multinomial Classifier | Alpha = 1          | 0.48               | 0.43          | 0.14                    |
| C-Support Vector Classification    | C = 1              | 0.26               | 0.26          | 0.03                    |

## Implementation 

> To train the models, the train_test_split function was used from sklearn, dividing the data into 90% training and 10% testing. Each model was trained using the fit function from sklearn. After the algorithm was trained using x_train and y_train, the predict function was used to predict y_values from the x_train. Then y_values were compared to y_train to assess if there was overfitting. This process yielded the column Training Accuracy. 

>Next, the predict function was used to forecast y_values from x_test. This will subsequently be compared with y_test to yield a classification report, where the Test accuracy and Test Macro F-1 were obtained. The macro average F-1 score was chosen over the weighted average given the contexts of unbalanced data. This way results can demonstrate if underrepresented classes are being adequately classified. This way it can be assess if a classifier was doing a good job sorting classes, rather than putting them all in one class.  

## Analysis

>Exclusively assessing the Test Accuracy and Test Weighted F-1 Score, the Random Forest Classifier and the Naïve Bayes Multinomial Classifier obtain the best scores during this evaluation. The lowest scores were from the Support Vector Classifier, followed by the K-Neighbors Classifier. This would be true if assessing only the measurements of the classification report by sklearn. However, for a better peek at what is happening within each algorithm, the training accuracy was retrieved and reported. It demonstrated that the Random Forest algorithm was overfitting the data, while Naïve Bayes did a better job generalizing the data.  On the lower performance side, the Support Vector Classifier generalized too much the data, as well as the K-Neighbors classifier. Random Forest is known to be data-intensive, and even with more than 60k instances, the model did overfit with the hyperparameters used in this research. Further assessment may be done with a Random Forest Classifier by tunning different hyperparameters and limiting the ability of the algorithm to overfit (e.g. max_depth). From the results presented, the Naïve Bayes algorithm with an alpha of 0.1 seems to be performing best at classifying this task.  The complete results of the classification report are below in figure 2.

### Figure 2

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| AP           | 0.48      | 0.20   | 0.28     | 346     |
| Atlantic     | 0.00      | 0.00   | 0.00     | 13      |
| CBC          | 0.50      | 0.04   | 0.07     | 28      |
| CBS          | 0.26      | 0.14   | 0.18     | 388     |
| Fox          | 0.45      | 0.32   | 0.37     | 443     |
| HuffingtonUK | 0.38      | 0.12   | 0.18     | 350     |
| HuffingtonUS | 0.51      | 0.78   | 0.61     | 1604    |
| LATimes      | 0.00      | 0.00   | 0.00     | 60      |
| Skynews      | 0.50      | 0.03   | 0.05     | 145     |
| WP           | 0.41      | 0.37   | 0.39     | 854     |
| bbc          | 0.52      | 0.30   | 0.38     | 410     |
| guardian     | 0.52      | 0.77   | 0.62     | 999     |
| telegraph    | 0.52      | 0.37   | 0.43     | 440     |
| accuracy     |           |        | 0.48     | 6080    |
| macro avg    | 0.39      | 0.26   | 0.27     | 6080    |
| weighted avg | 0.46      | 0.48   | 0.44     | 6080    |


>Almost all algorithms did poorly to classify classes that had very little data. 'LATimes' 'CBC' and 'Atlantic' regularly scored zero for f-1 score. Only the Random Forest classifed a few correct instances here. Conversely, 'HuffingtonUS' and 'guardian' had better f-1 scores, as illustrated in figure 2. There may be a correlation between the amount of data and the ability of the classifier algorithms to perform. Then this task is not hindered by the capabilities of the algorithm to classify newspaper titles but by the size of the data input that they receive. 

## Conclusion 

>It may well be that Naïve Bayes, since it is a probabilistic classifier, could better weight proper nouns and entities that were related to each newspaper's domain, and therefore classify them better. Therfore, becoming the best performer amogst the algorithms implemented in this reserach.
>Unbalanced data does negativly affect the capabilties of these algorithms to properly classify instances, and underpresented classes will score close to or zero no matter what algorithm you use. To completly judge how each algorithm performed at this task, a more balanced data set would be ideal to view how they handle the classification. At last, the results are encouraging to say that an algorithm can classify news titles by issuer. 


