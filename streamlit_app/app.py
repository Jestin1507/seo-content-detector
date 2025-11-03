import streamlit as st
# Note: We import 'utils' functions from the 'utils' folder
from utils.scorer import load_assets, analyze_content

# Set page config for a wider layout
st.set_page_config(page_title="SEO Content Analyzer", layout="wide")

st.title("🚀 SEO Content Quality & Duplicate Detector")
st.write("Enter a URL to analyze its content quality, readability, and check for duplicates against our dataset.")

# --- Load Models ---
model, vectorizer, tfidf_matrix, url_list = load_assets()

# --- URL Input ---
url_to_analyze = st.text_input("Enter URL to analyze:", placeholder="https://example.com/blog/my-article")

if st.button("Analyze Content"):
    if url_to_analyze and model:
        with st.spinner(f"Analyzing {url_to_analyze}..."):
            try:
                result = analyze_content(url_to_analyze, model, vectorizer, tfidf_matrix, url_list)
                
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success(f"Analysis Complete! Title: {result['title']}")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Quality Label", result['quality_label'])
                    
                    thin_note = " (Thin Content)" if result['is_thin'] else ""
                    col2.metric("Word Count", f"{result['word_count']}{thin_note}")
                    
                    col3.metric("Readability Score", f"{result['readability']:.2f}")

                    st.subheader("Duplicate Content Check")
                    if result['similar_to']:
                        st.warning("⚠️ Potential Duplicate Found!")
                        for item in result['similar_to']:
                            st.write(f"**Similarity:** {item['similarity'] * 100:.1f}%")
                            st.write(f"**Matching URL:** {item['url']}")
                    else:
                        st.info("✅ No significant duplicates found in our dataset.")
                    
                    with st.expander("Show Full Analysis (JSON)"):
                        st.json(result)

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
    
    elif not model:
        st.error("Assets not loaded. Please run the notebook and check file paths.")
    else:
        st.warning("Please enter a URL to analyze.")