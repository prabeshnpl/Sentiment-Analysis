from flask import Flask, render_template, request
import joblib
from utils.modules.preprocessing_v1 import clean_and_lemmatize

NB_model = joblib.load('utils/models/v7/sentiment_detection_nb_model_v7.pkl')
SVM_model = joblib.load('utils/models/v7/sentiment_detection_svm_model_v7.pkl')
tf_idf = joblib.load('utils/models/v7/tfidf_vectorizer_v7.pkl')

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():

    nb_prediction = None
    svm_prediction = None
    input_text = ''
    nb_prediction_probs = ''
    svm_prediction_scores = ''

    if request.method == 'POST':

        # Get user input
        input_text = request.form.get('input_text', '')

        # Clean the sentence
        processed_text = clean_and_lemmatize(input_text)

        # Transform into digits
        X = tf_idf.transform(processed_text)

        # Predict using Naive Bayes' model
        nb_prediction = NB_model.predict(X)
        nb_prediction_probs = NB_model.predict_proba(X)[0]
        print(f"Negative :{nb_prediction_probs[0]}\nNeutral : {nb_prediction_probs[1]}\nPositive: {nb_prediction_probs[2]}")

        # Predict using SVM model
        svm_prediction = SVM_model.predict(X)
        svm_prediction_scores = SVM_model.decision_function(X)[0]
        print(f"Negative :{svm_prediction_scores[0]}\nNeutral : {svm_prediction_scores[1]}\nPositive: {svm_prediction_scores[2]}")

        # Convert output
        label_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}
        nb_prediction = label_map[int(nb_prediction[0])]
        svm_prediction = label_map[int(svm_prediction[0])]

    return render_template(
        'index.html', 
        input_text=input_text,
        nb_prediction=nb_prediction,
        nb_probability=nb_prediction_probs,
        svm_prediction=svm_prediction,
        svm_prediction_scores=svm_prediction_scores
    )

if __name__ == '__main__':
    app.run(debug=True)