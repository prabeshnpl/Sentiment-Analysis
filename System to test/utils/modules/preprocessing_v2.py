import re
import emoji

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

# Slang translator
def translate_slang(text):
    words = text.split()
    # Replace the word if it exists in our dictionary, otherwise keep the word
    new_words = [slang_dict[w] if w in slang_dict else w for w in words]
    return " ".join(new_words)
    
# Define a Lightweight NLTK Cleaning Function
def clean_texts(raw_text):
    """Performs basic cleaning."""

    if not isinstance(raw_text, str):
        return ""

    # Remove unnecessary whitespaces and make it lowercase.
    raw_text = (" ".join(raw_text.split())).lower()

    # Url remover
    raw_text = re.sub(r'http\S+|www\S+|https\S+', '', raw_text, flags=re.MULTILINE)

    # Remove words like "soooooooooooooooo" that contains more than two times of repeating letter in a word
    raw_text = re.sub(r'(.)\1+', r'\1\1', raw_text)

    # Remove emojis
    raw_text = emoji.demojize(raw_text, delimiters=(" ", " "))

    # Translate slang into proper words
    raw_text = translate_slang(raw_text)

    return raw_text
