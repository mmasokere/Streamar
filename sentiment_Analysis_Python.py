import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.naive_bayes import GaussianNB
train_data = pd.read_csv("trying.csv")

# print(train_data.info())
# print(train_data['label'].value_counts())
# print(train_data.head(5)['tweet'])


def process_tweet(tweet):
    return " ".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", "", tweet.lower()).split())


train_data['processed_tweets'] = train_data['tweets'].apply(process_tweet)
print(train_data.head())

x_train, x_test, y_train, y_test = train_test_split(
    train_data["processed_tweets"], train_data["sentiment"], test_size=0.2, random_state=42)
count_vect = CountVectorizer(stop_words='english')
transformer = TfidfTransformer(norm='l2', sublinear_tf=True)
x_train_counts = count_vect.fit_transform(x_train)
x_train_tfidf = transformer.fit_transform(x_train_counts)
x_test_counts = count_vect.transform(x_test)
x_test_tfidf = transformer.transform(x_test_counts)
# print(x_train_counts.shape)
# print(x_train_tfidf.shape)
# print(x_test_counts.shape)
# print(x_test_tfidf.shape)
model = RandomForestClassifier(n_estimators=500)
model.fit(x_train_tfidf, y_train)
predictions = model.predict(x_test_tfidf)

model2 = GaussianNB()
model2.fit(x_train_tfidf, y_train)
predictions2 = model2.predict(x_test_tfidf)

print(confusion_matrix(y_test, predictions))
print(f1_score(y_test, predictions))
# print(confusion_matrix(x_test, predictions))
# print(f1_score(x_test, predictions))


# def clean_tweet(tweet):
#     return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

# print(train_data.head())
# drop_features(['id','tweet'],train_data)
# print(train_data.info())
