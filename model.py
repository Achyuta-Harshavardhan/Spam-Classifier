import pandas as pd
import numpy as np
import pickle
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,accuracy_score

df = pd.read_csv('spam.csv',encoding='latin-1')
x = df.iloc[:,1]
y = df.iloc[:,0]

ps = PorterStemmer()
corpus = []
for i in range(0, len(x)):
    review = re.sub('[^a-zA-Z]', ' ', x[i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

tf = TfidfVectorizer(max_features=5000,ngram_range=(1,3))
x = tf.fit_transform(corpus).toarray()

pickle.dump(tf,open('tfidf transform.pkl','wb'))

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

classifier=MultinomialNB()
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
score = confusion_matrix(y_test, pred)
#print(score)
score = accuracy_score(y_test, pred)
#print(score)

filename = 'Spam Classifier.pkl'
pickle.dump(classifier, open(filename, 'wb'))