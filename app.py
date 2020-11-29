# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'Spam Classifier.pkl'
classifier = pickle.load(open(filename, 'rb'))
tf = pickle.load(open('tfidf transform.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        vect = tf.transform([message]).toarray()
        my_prediction = classifier.predict(vect)
        if my_prediction in ['ham']:
            my_prediction = 0
        else:
            my_prediction = 1
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)