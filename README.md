# Model Training & Performance

The core of this application is a Sentiment Analysis engine trained on the Amazon Fine Food Reviews dataset.

### 🛠️ Data Preprocessing Pipeline
To ensure high-quality input, the following NLP pipeline was implemented:
- **Noise Removal:** Stripped URLs, repetitive characters (e.g., "sooooooo" -> "soo"), and unnecessary whitespace.
- **Stop word Removal**: Removed words that have no weight in training the model such as 'is', 'the', etc.
- **Emoji Handling:** Converted emojis to text descriptions to capture emotional context.
- **Negation Preservation:** Specifically excluded negation words (not, no, never) from the stop-word removal list to maintain sentiment integrity.
- **Lemmatization:** Used `spaCy` batch processing to reduce words to their root forms.

### 📊 Model Comparison
We compared two popular classification algorithms using a 5-fold Cross-Validation Grid Search:

| Model | Accuracy | Macro F1-Score | Key Advantage |
| :--- | :--- | :--- | :--- |
| **Linear SVM** | **87%** | **0.75** | High precision across all three classes (Pos/Neg/Neu). |
| **Naive Bayes**| 78% | 0.66 | Extremely fast training and low computational cost. |

### 💾 Model Artifacts
The following files are generated and used by the Flask application:
- `sentiment_detection_svm_model_v5.pkl`: The primary trained SVM classifier.
- `sentiment_detection_nb_model_v5.pkl`: The primary trained Naive Bayes classifier.
- `tfidf_vectorizer_v5.pkl`: The vectorizer containing the 5,000 most significant word features.

## 🚀 How to Re-train
If you wish to update the model with new data:
1. Place your CSV in `datasets/amazon_training.csv`.
2. Run the training notebook or script.
3. The script will automatically export the updated `.pkl` files for the Flask app.

# How to run Flask Sentiment Analysis App

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

* Python 3.8 or higher
* `pip` (Python package installer)

### 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/prabeshnpl/Sentiment-Analysis.git

   cd '.\System to test'

2. **It is recommended to use a virtual environment to avoid dependency conflicts.**
   ```bash
   python -m venv venv

   .\venv\Scripts\activate

3. **Install requirements.txt**
    ```bash
    pip install -r requirements.txt

4. **Run the app**
    ```bash
    flask run --debug

5. **Access the App**
    - Open your web browser and navigate to: ```http://127.0.0.1:5000```
