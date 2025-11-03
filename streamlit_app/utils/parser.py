import re
from bs4 import BeautifulSoup

def parse_html(html_content):
    """
    Parses raw HTML content to extract title, clean text, and word count.
    """
    try:
        if not isinstance(html_content, str):
            return None, None, 0
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract title
        title = soup.title.string.strip() if soup.title else 'No Title'
        
        # Get all text and clean it
        text = soup.get_text(separator=' ', strip=True)
        text = re.sub(r'\s+', ' ', text).strip()
        
        word_count = len(text.split())
        
        return title, text, word_count
    except Exception as e:
        print(f"Error parsing HTML: {e}")
        return None, None, 0