import requests
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import os

# Import our custom modules
from utils.parser import parse_html
from utils.features import extract_all_features

# --- CRITICAL: Build paths relative to THIS file ---
# This file is in: seo-content-detector/streamlit_app/utils/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to models: ../models/
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')

# Path to data: ../../data/
DATA_DIR = os.path.join(BASE_DIR, '..', '..', 'data')

# Define full paths to assets
MODEL_PATH = os.path.join(MODELS_DIR, 'quality_model.pkl')
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
MATRIX_PATH = os.path.join(MODELS_DIR, 'full_tfidf_matrix.pkl')
URL_LIST_PATH = os.path.join(DATA_DIR, 'processed_urls.csv')

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

@st.cache_resource
def load_assets():
    """
    Loads all necessary models and data assets.
    Uses st.cache_resource to load only once.
    """
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        tfidf_matrix = joblib.load(MATRIX_PATH)
        url_list = pd.read_csv(URL_LIST_PATH)['url']
        
        return model, vectorizer, tfidf_matrix, url_list
    except FileNotFoundError as e:
        st.error(f"Error loading assets: {e}. Did you run the notebook to generate assets?")
        return None, None, None, None

def scrape_url(url):
    """
    Scrapes a single URL and returns its HTML content.
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status() # Raise an error for bad status codes
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return None

def analyze_content(url, model, vectorizer, tfidf_matrix, url_list):
    """
    The main analysis pipeline for a single URL.
    """
    html_content = scrape_url(url)
    if not html_content:
        return {"error": "Failed to fetch URL. It might be down or blocking scrapers."}

    title, text, word_count = parse_html(html_content)
    if word_count == 0:
        return {"error": "Could not extract readable content from the page."}
    
    features = extract_all_features(text)
    
    features_for_model = pd.DataFrame({
        'word_count': [features['word_count']],
        'sentence_count': [features['sentence_count']],
        'flesch_reading_ease': [features['flesch_reading_ease']]
    })
    quality_label = model.predict(features_for_model)[0]

    new_tfidf = vectorizer.transform([features['clean_text']])
    sim_scores = cosine_similarity(new_tfidf, tfidf_matrix).flatten()
    
    top_match_index = sim_scores.argsort()[-1]
    top_similarity = sim_scores[top_match_index]
    
    similar_to = []
    if top_similarity > 0.80:
         similar_to.append({
             "url": url_list.iloc[top_match_index],
             "similarity": round(float(top_similarity), 2)
         })
    
    result = {
        "url": url,
        "title": title,
        "word_count": int(features['word_count']),
        "readability": round(float(features['flesch_reading_ease']), 2),
        "quality_label": quality_label,
        "is_thin": bool(features['word_count'] < 500),
        "similar_to": similar_to
    }
    return result