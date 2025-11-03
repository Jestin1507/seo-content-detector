# streamlit_app/utils/nltk_setup.py
import os
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer
import ssl

def ensure_punkt_tab_exists():
    """Ensure punkt tokenizer exists and is properly initialized"""
    setup_nltk()
    try:
        tokenizer = PunktSentenceTokenizer()
        # Test the tokenizer to ensure it's working
        test_text = "This is a test. This is another test."
        tokenizer.tokenize(test_text)
        return True
    except Exception as e:
        print(f"Error initializing PunktSentenceTokenizer: {str(e)}")
        return False

def setup_nltk():
    """Initialize NLTK with required resources"""
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # Create a local nltk_data directory in the project root
    nltk_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'nltk_data')
    os.makedirs(nltk_data_path, exist_ok=True)

    # Add the path to NLTK's search paths
    if nltk_data_path not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data_path)  # Make our path the first one to check

    # Download all required NLTK data
    required_resources = [
        'punkt',
        'stopwords',
        'averaged_perceptron_tagger'
    ]

    for resource in required_resources:
        try:
            nltk.download(resource, download_dir=nltk_data_path, quiet=True)
        except Exception as e:
            print(f"Error downloading {resource}: {str(e)}")

    # Initialize Punkt tokenizer directly
    try:
        tokenizer = PunktSentenceTokenizer()
    except Exception as e:
        print(f"Warning: Could not initialize PunktSentenceTokenizer: {str(e)}")

    return nltk_data_path