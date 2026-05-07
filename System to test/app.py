from flask import Flask, render_template, request
import joblib
from utils.modules.preprocessing_v2 import clean_texts

NB_model = joblib.load('utils/models/v5/sentiment_detection_nb_model_v5.pkl')
SVM_model = joblib.load('utils/models/v5/sentiment_detection_svm_model_v5.pkl')
tf_idf = joblib.load('utils/models/v5/tfidf_vectorizer_v5.pkl')

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():

    nb_prediction = None
    svm_prediction = None
    input_text = ''

    if request.method == 'POST':

        # Get user input
        input_text = request.form.get('input_text', '')

        # Clean the sentence
        processed_text = clean_texts(input_text)

        # Transform into digits
        X = tf_idf.transform([processed_text])
        
        # Predict using Naive Bayes' model
        nb_prediction = NB_model.predict(X)

        # Predict using SVM model
        svm_prediction = SVM_model.predict(X)

        # Convert output
        label_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}
        nb_prediction = label_map[int(nb_prediction[0])]
        svm_prediction = label_map[int(svm_prediction[0])]

    return render_template(
        'index.html', 
        input_text=input_text,
        nb_prediction=nb_prediction,
        svm_prediction=svm_prediction
    )

if __name__ == '__main__':
    app.run(debug=True)