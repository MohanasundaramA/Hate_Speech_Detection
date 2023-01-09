# **Hate Speech Detection**

## Abstract
Hate speech is extremely common on platforms such as Twitter, Facebook, comments sections and even blogs or biased online publications, and even though things like profanity filters exist, they only filter out obscenities and swear words.We find that the vast majority of hate speech consists of veiled attacks, or otherwise uses words that would in other contexts be completely innocuous but are being used to attack an individual or group. Most of even this is contextualized, so to know the context of each of the sentences and then gauge the hatefulness to a near perfect accuracy would require some knowledge of the topic under discussion which is beyond the scope of our rule-based algorithm. However, we find we can achieve a healthy precision in not just the detection of hate speech, but also its segregation into weakly or strongly hateful speech.

## Requirements
- Python - 3.6.7
- Numpy - 1.16.4
- nltk - 3.2.5
- Matplotlib - 3.0.3
- Sklearn - 1.2.0

## Dataset
Dataset can be downloaded by clicking [here](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset). It contains 7 columns namely count, hate_speech, offensive_language, neither, class, tweet

Tweets are classified as: hate-speech(1430), offensive language(19190), and neither(4163)

## Flow of the project
- Pre-processing the tweets (Remove: stop words, emojis, mentions, urls and all kind of noise, along with a stage of lemmatizing and stemming)
- Use TF-IDF vectorizer to convert the data into a model of numerical features that are ready to be used for classification
- Apply cross validation on the training vectors with 0.2 splitting factor, while tuning some of the selected parameters to enhance the accuracy
- Use the best estimator of selected classifier to predict the test labels

## Classifiers
- Logistic Regression
  + Accuracy = 89.75%
- Random Forest Classifier
  + Accuracy = 89.75%
- Linear Support Vector Classifier
  + Accuracy = 89.33%
- AdaBoost Classifier
  + Accuracy = 93.56%
- XGBoost Classifier
  + Accuracy = 89.79%

## Results
With the balanced dataset, at a baseline accuracy of 78%, the best models (Random Forest, Logistic Regression, and SVM) improved accuracy to around 92%. In terms of other metrics, they exhibited the highest F1 scores at around 87-88%, but do not have the highest recall or precision scores. They also have the highest ROC-AUC scores at around 96% and PR-AUC scores at around 95%.
