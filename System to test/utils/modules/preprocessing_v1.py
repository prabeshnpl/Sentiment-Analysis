import re
import nltk
import emoji
from nltk.corpus import stopwords
import spacy
from nltk.stem import WordNetLemmatizer

# Preprocessing :
# Whitespace Normalization -> Lowercasing -> Tokenization -> Stop-word Removal -> Lemmatization

# Define a dictionary of common Twitter slang
slang_dict = {
    "lol": "laughing out loud",
    "idk": "i do not know",
    "smh": "shaking my head",
    "brb": "be right back",
    "omg": "oh my god",
    "rt": "retweet",
    "u": "you",
    "r": "are",
    "bday": "birthday"
}

# Setup Resources
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

negation_words = {'not', 'no', 'never', 'neither', 'nor', 'none', 'n\'t', 'nt', 'barely', 'hardly'}
stop_words = set(stopwords.words('english')) - negation_words  # Preserve negation words

# Load spaCy if available; otherwise use NLTK fallback for lemmatization.
_use_spacy = True
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    nlp = None
    _use_spacy = False
    lemmatizer = WordNetLemmatizer()

    # Log fallback to NLTK lemmatizer during import time if spacy model is unavailable.
    print("Warning: spaCy model 'en_core_web_sm' not found. Falling back to NLTK lemmatizer.")


def _lemmatize_with_nltk(text):
    tokens = text.split()
    return " ".join(lemmatizer.lemmatize(token, pos='v') for token in tokens)

# Slang translator
def translate_slang(text):
    words = text.split()
    # Replace the word if it exists in our dictionary, otherwise keep the word
    new_words = [slang_dict[w] if w in slang_dict else w for w in words]
    return " ".join(new_words)
    
# Define a Lightweight NLTK Cleaning Function
def clean_and_tokenize(raw_text):
    """Performs basic NLTK cleaning before spaCy lemmatization."""
    if not isinstance(raw_text, str):
        return ""

    # Remove unnecessary whitespaces and make it lowercase.
    text = (" ".join(raw_text.split())).lower()

    # Url remover
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove words like "soooooooooooooooo" that contains more than two times of repeating letter in a word
    text = re.sub(r'(.)\1+', r'\1\1', text)

    # Remove emojis
    text = emoji.demojize(text, delimiters=(" ", " "))

    # Translate slang into proper words
    text = translate_slang(text)
    
    # Convert sentence into tokens
    tokens = nltk.word_tokenize(text)

    # Return only alphabetic words
    return " ".join([w for w in tokens if w not in stop_words])

# Define a Function for Optimized spaCy Lemmatization using nlp.pipe (Batching)
def clean_and_lemmatize(raw_texts, batch_size:int=1000):
    """Performs lemmatization over cleaned texts(from 'clean_and_tokenize' function)."""
    
    final_preprocessed = []
    
    if isinstance(raw_texts, str):
        raw_texts = [raw_texts]
        
    raw_cleaned_texts = [clean_and_tokenize(t) for t in raw_texts]
    
    if _use_spacy and nlp is not None:
        for doc in nlp.pipe(raw_cleaned_texts, batch_size=batch_size):
            final_preprocessed.append(" ".join([token.lemma_ for token in doc]))
    else:
        for text in raw_cleaned_texts:
            final_preprocessed.append(_lemmatize_with_nltk(text))
        
    return final_preprocessed

