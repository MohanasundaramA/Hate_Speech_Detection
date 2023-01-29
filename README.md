# **Hate Speech Detection**

## Abstract
Hate speech is extremely common on platforms such as Twitter, Facebook, comment sections, and even blogs or biased online publications, and even though things like profanity filters exist, they only filter out obscenities and swear words. We find that the vast majority of hate speech consists of veiled attacks or otherwise uses words that would in other contexts be completely innocuous but are being used to attack an individual or group. Most of even this is contextualized, so to know the context of each of the sentences and then gauge the hatefulness with near-perfect accuracy would require some knowledge of the topic under discussion, which is beyond the scope of our rule-based algorithm. However, we discover a model to distinguish between speech that is mildly or intensely hostile and identify it with a fair degree of precision.

## Requirements
- Python - 3.6.7
- Numpy - 1.16.4
- nltk - 3.2.5
- Matplotlib - 3.0.3
- Sklearn - 1.2.0

## Dataset
Use the link below to get the dataset. The Davidson Dataset comprises 24802 tweets. Utilizing crowdsourcing to identify the tweets, 5.77% or 1430 rows were labelled as "hate," 77.43% or 19190 rows were labelled as "offensive," and 16.80% or 4163 rows were labelled as "neither." It has 7 columns, including "tweet," "class," "neither," "hate speech" and "bad language".

## Flow of the project
1. Tweet pre-processing (removal of stop words, emojis, mentions, urls, and other noise, as well as a lemmatizing and stemming stage).
2. Use the  TF-IDF vectorizer to convert the data into a model of numerical features that are ready to be used for classification.
3. Train five different ML classifiers using the training vectors with a splitting factor of 0.2.
4. Tune some of the parameters of the best classifier among five to improve the model's accuracy.
5. Use the best estimator of the selected classifier to predict the test labels.

## Preprocessing Techniques:
- Letter casing
- Tokenizing
- Noise removal
- Stopword removal
- Normalization
- Stemming or Lemmatization
- Vectorization

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
  
After tuning the best model (AdaBoost), model accuracy increases to 95.6%.

## Result
Different machine learning models have different strengths that make some better than others for certain tasks, such as detecting hate speech. Some models are more accurate, while others are more efficient. It is important to use different models and compare their performances in order to find the best one for hate speech detection. The model we have trained is a little overfit to the training data, but we can handle this by using different regularisation techniques. But still, we had achieved 95% accuracy on the validation data.
