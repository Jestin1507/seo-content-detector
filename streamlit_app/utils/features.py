import re
import os
import textstat
from nltk.tokenize import PunktSentenceTokenizer
from utils.nltk_setup import ensure_punkt_tab_exists

# Get PunktSentenceTokenizer instance after ensuring punkt_tab exists
ensure_punkt_tab_exists()
sentence_tokenizer = PunktSentenceTokenizer()

def clean_text(text):
    """
    Cleans text by lowercasing and removing non-alphabetic characters.
    """
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text) # Keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def safe_flesch_score(text):
    """
    Calculates Flesch score safely, truncating text and handling errors.
    """
    try:
        # TRUNCATE text to first 50,000 chars
        return textstat.flesch_reading_ease(str(text)[:50000])
    except Exception:
        return 0.0

def extract_all_features(text):
    """
    Extracts all necessary features from raw text for scoring.
    """
    if not text:
        return None

    word_count = len(text.split())
    sentence_count = len(sentence_tokenizer.tokenize(str(text)))
    flesch_score = safe_flesch_score(text)
    cleaned_text_content = clean_text(text)
    
    features = {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'flesch_reading_ease': flesch_score,
        'clean_text': cleaned_text_content
    }
    return features