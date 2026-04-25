"""
Zomato Restaurant Sentiment Analysis - Streamlit Web App
"""

import streamlit as st
import joblib
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

st.set_page_config(
    page_title="Zomato Sentiment Analyzer",
    page_icon="🍽️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def load_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet',   quiet=True)
    nltk.download('punkt',     quiet=True)
    nltk.download('punkt_tab', quiet=True)

load_nltk()

@st.cache_resource
def load_models():
    model   = joblib.load('sentiment_model.pkl')
    tfidf   = joblib.load('tfidf_vectorizer.pkl')
    encoder = joblib.load('label_encoder.pkl')
    return model, tfidf, encoder

try:
    model, tfidf, encoder = load_models()
    models_loaded = True
except FileNotFoundError:
    models_loaded = False

lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words('english'))

CONTRACTIONS = {
    "n't": " not", "'re": " are", "'s": " is", "'d": " would",
    "'ll": " will", "'ve": " have", "'m": " am"
}

def expand_contractions(text):
    for k, v in CONTRACTIONS.items():
        text = text.replace(k, v)
    return text

def preprocess_text(text):
    text = str(text).lower()
    text = expand_contractions(text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if len(t) > 2]
    return ' '.join(tokens)

def predict_sentiment(review_text):
    cleaned     = preprocess_text(review_text)
    vectorized  = tfidf.transform([cleaned])
    pred_label  = model.predict(vectorized)[0]
    proba       = model.predict_proba(vectorized)[0]
    label_name  = encoder.inverse_transform([pred_label])[0]
    class_probs = {encoder.classes_[i]: float(proba[i]) for i in range(len(proba))}
    return label_name, max(proba), class_probs

# ── Session state defaults ────────────────────────────────────────────────────
if "review_text" not in st.session_state:
    st.session_state.review_text = ""

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title  { font-size:2.4rem; font-weight:800; color:#e23744; text-align:center; margin-bottom:0.2rem; }
    .subtitle    { text-align:center; color:#666; font-size:1rem; margin-bottom:2rem; }
    .result-box  { padding:1.5rem; border-radius:12px; text-align:center; margin:1rem 0; }
    .positive    { background:#d4edda; border:2px solid #28a745; }
    .negative    { background:#f8d7da; border:2px solid #dc3545; }
    .neutral     { background:#fff3cd; border:2px solid #ffc107; }
    .result-label{ font-size:2rem; font-weight:700; margin-bottom:0.3rem; }
    .confidence  { font-size:1rem; color:#555; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🍽️ Zomato Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict customer sentiment from restaurant reviews using NLP & Machine Learning</div>', unsafe_allow_html=True)

if not models_loaded:
    st.error("⚠️ Model files not found! Make sure sentiment_model.pkl, tfidf_vectorizer.pkl, and label_encoder.pkl are in the repo.")
    st.stop()

# ── Example buttons ───────────────────────────────────────────────────────────
EXAMPLES = {
    "pos": "The food was absolutely amazing and the service was top notch! Great ambience too.",
    "neu": "It was okay, nothing special but not bad either. Average food and service.",
    "neg": "Terrible experience. Food was cold and staff was rude. Will never come back."
}

st.subheader("📝 Enter a Restaurant Review")
st.caption("Try an example or type your own review below:")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("😊 Try Positive", use_container_width=True):
        st.session_state.review_text = EXAMPLES["pos"]
        st.rerun()
with col2:
    if st.button("😐 Try Neutral", use_container_width=True):
        st.session_state.review_text = EXAMPLES["neu"]
        st.rerun()
with col3:
    if st.button("😞 Try Negative", use_container_width=True):
        st.session_state.review_text = EXAMPLES["neg"]
        st.rerun()

# ── Text area ─────────────────────────────────────────────────────────────────
review_input = st.text_area(
    "Review Text",
    value=st.session_state.review_text,
    placeholder="e.g. The biryani was incredible and the staff was very friendly...",
    height=120,
    label_visibility="collapsed"
)

# Keep session state in sync if user types manually
st.session_state.review_text = review_input

# ── Analyze button ────────────────────────────────────────────────────────────
if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
    if not review_input.strip():
        st.warning("Please enter a review to analyze.")
    else:
        with st.spinner("Analyzing..."):
            sentiment, confidence, class_probs = predict_sentiment(review_input)

        emoji_map   = {"Positive": "😊", "Neutral": "😐", "Negative": "😞"}
        color_class = sentiment.lower()
        emoji       = emoji_map.get(sentiment, "🤔")

        st.markdown(f"""
        <div class="result-box {color_class}">
            <div class="result-label">{emoji} {sentiment}</div>
            <div class="confidence">Confidence: {confidence:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("📊 Confidence Breakdown")
        order  = ["Positive", "Neutral", "Negative"]
        emojis = ["😊", "😐", "😞"]

        cols = st.columns(3)
        for i, (label, em) in enumerate(zip(order, emojis)):
            if label in class_probs:
                with cols[i]:
                    st.metric(label=f"{em} {label}", value=f"{class_probs[label]:.1%}")

        for label, em in zip(order, emojis):
            if label in class_probs:
                st.markdown(f"**{em} {label}**")
                st.progress(class_probs[label])

# ── Batch Analysis ────────────────────────────────────────────────────────────
st.divider()
st.subheader("📋 Batch Analysis")
st.markdown("Analyze multiple reviews at once — one review per line.")

batch_input = st.text_area(
    "Paste multiple reviews",
    placeholder="The food was amazing\nService was terrible\nIt was just okay",
    height=150,
    label_visibility="collapsed"
)

if st.button("🔍 Analyze All Reviews", use_container_width=True):
    lines = [l.strip() for l in batch_input.strip().split('\n') if l.strip()]
    if not lines:
        st.warning("Please enter at least one review.")
    else:
        results  = []
        progress = st.progress(0)
        for i, line in enumerate(lines):
            sent, conf, _ = predict_sentiment(line)
            results.append({
                "Review":     line[:80] + ("..." if len(line) > 80 else ""),
                "Sentiment":  sent,
                "Confidence": f"{conf:.1%}"
            })
            progress.progress((i + 1) / len(lines))

        df_results = pd.DataFrame(results)

        def color_sentiment(val):
            return {
                "Positive": "background-color: #d4edda",
                "Neutral":  "background-color: #fff3cd",
                "Negative": "background-color: #f8d7da"
            }.get(val, "")

        st.dataframe(
            df_results.style.map(color_sentiment, subset=['Sentiment']),
            use_container_width=True,
            hide_index=True
        )

        counts = df_results['Sentiment'].value_counts()
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("😊 Positive", counts.get("Positive", 0))
        with c2: st.metric("😐 Neutral",  counts.get("Neutral",  0))
        with c3: st.metric("😞 Negative", counts.get("Negative", 0))

# ── About ─────────────────────────────────────────────────────────────────────
with st.expander("ℹ️ About this App"):
    st.markdown("""
    **Model Details**
    - **Algorithm**: Multinomial Naive Bayes (tuned with GridSearchCV)
    - **Text Features**: TF-IDF Vectorizer (5000 features, bigrams)
    - **Preprocessing**: Contraction expansion → Lowercase → Punctuation removal → Stopword removal → Lemmatization
    - **Classes**: Positive (Rating ≥ 4) | Neutral (3–3.5) | Negative (< 3)

    **Dataset**
    - Source: Zomato restaurant reviews (~10,000 reviews, 105 restaurants)

    **Tech Stack**
    - Python · scikit-learn · NLTK · Streamlit · joblib
    """)
