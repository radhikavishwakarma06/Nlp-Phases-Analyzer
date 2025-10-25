import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import csv
from urllib.parse import urljoin
import time
import random
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from textblob import TextBlob
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import io
import os # <-- Added os import for file existence check

# ---------------------------
# Configuration & constants
# ---------------------------
SCRAPED_DATA_PATH = 'politifact_data.csv'
N_SPLITS = 5
STOP_WORDS_SET = STOP_WORDS
PRAGMATIC_TERMS = ["must", "should", "might", "could", "will", "?", "!"]

# ---------------------------
# SpaCy loader (robust for cloud)
# ---------------------------
@st.cache_resource
def get_spacy_model():
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError as exc:
        st.error("SpaCy 'en_core_web_sm' model not found. Add the wheel URL for the model in requirements.txt.")
        st.code("""
# Add to requirements.txt (example):
https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
imbalanced-learn
        """, language='text')
        raise exc

try:
    NLP = get_spacy_model()
except Exception:
    st.stop()

# ---------------------------
# 1) Data collection / scraping
# ---------------------------
# Note: This function cannot be cached as it performs network I/O and modifies the state/filesystem
def collect_claims_within_range(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """
    Crawl Politifact listing pages between start_date and end_date and return DataFrame
    with columns: author, statement, source, date, label.
    """
    base_url = "https://www.politifact.com/factchecks/list/"
    cur_url = base_url
    outbuf = io.StringIO()
    writer = csv.writer(outbuf)
    writer.writerow(["author", "statement", "source", "date", "label"])
    total_added = 0
    page_num = 0

    status_placeholder = st.empty()
    status_placeholder.caption(f"Scraping claims from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    while cur_url and page_num < 200:
        page_num += 1
        status_placeholder.text(f"Fetching page {page_num} — collected {total_added} claims so far.")
        try:
            resp = requests.get(cur_url, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
        except requests.RequestException as e:
            status_placeholder.error(f"Network error: {e}. Aborting.")
            break

        to_write = []
        for item in soup.find_all("li", class_="o-listicle__item"):
            desc = item.find("div", class_="m-statement__desc")
            date_text = desc.get_text(strip=True) if desc else None
            claim_date = None
            if date_text:
                m = re.search(r"stated on ([A-Za-z]+\s+\d{1,2},\s+\d{4})", date_text)
                if m:
                    try:
                        claim_date = pd.to_datetime(m.group(1), format='%B %d, %Y')
                    except ValueError:
                        continue

            if claim_date:
                if start_date <= claim_date <= end_date:
                    quote_block = item.find("div", class_="m-statement__quote")
                    statement = quote_block.find("a", href=True).get_text(strip=True) if quote_block and quote_block.find("a", href=True) else None
                    source_a = item.find("a", class_="m-statement__name")
                    source = source_a.get_text(strip=True) if source_a else None
                    footer = item.find("footer", class_="m-statement__footer")
                    author = None
                    if footer:
                        author_match = re.search(r"By\s+([^•]+)", footer.get_text(strip=True))
                        if author_match:
                            author = author_match.group(1).strip()
                    label_img = item.find("img", alt=True)
                    label = label_img['alt'].replace('-', ' ').title() if label_img and 'alt' in label_img.attrs else None
                    to_write.append([author, statement, source, claim_date.strftime('%Y-%m-%d'), label])
                elif claim_date < start_date:
                    status_placeholder.warning("Reached claims older than start date — stopping crawl.")
                    cur_url = None
                    break

        if cur_url is None:
            break

        writer.writerows(to_write)
        total_added += len(to_write)

        next_link = soup.find("a", class_="c-button c-button--hollow", string=re.compile(r"Next", re.I))
        if next_link and 'href' in next_link.attrs:
            cur_url = urljoin(base_url, next_link['href'])
        else:
            status_placeholder.success("No more pages found.")
            cur_url = None

    outbuf.seek(0)
    df = pd.read_csv(outbuf, header=0, keep_default_na=False)
    df = df.dropna(subset=['statement', 'label'])
    df.to_csv(SCRAPED_DATA_PATH, index=False)
    return df

# ---------------------------
# 2) Feature extraction helpers (Now cached)
# ---------------------------
@st.cache_data
def lexicon_clean(text: str) -> str:
    doc = NLP(text.lower())
    tokens = [tok.lemma_ for tok in doc if tok.text not in STOP_WORDS_SET and tok.is_alpha]
    return " ".join(tokens)

@st.cache_data
def pos_sequence(text: str) -> str:
    doc = NLP(text)
    return " ".join([tok.pos_ for tok in doc])

@st.cache_data
def sentiment_feats(text: str):
    tb = TextBlob(text)
    return [tb.sentiment.polarity, tb.sentiment.subjectivity]

@st.cache_data
def discourse_summary(text: str):
    doc = NLP(text)
    sents = [s.text.strip() for s in doc.sents]
    first_words = " ".join([s.split()[0].lower() for s in sents if len(s.split()) > 0])
    return f"{len(sents)} {first_words}"

@st.cache_data
def pragmatic_counts(text: str):
    t = text.lower()
    return [t.count(w) for w in PRAGMATIC_TERMS]

@st.cache_data(show_spinner=False)
def extract_features_series(series: pd.Series, phase: str, vectorizer=None):
    """
    Return X_features and fitted vectorizer (if used).
    Phases: 'Lexical & Morphological', 'Syntactic', 'Semantic', 'Discourse', 'Pragmatic'
    Uses st.cache_data to speed up re-runs.
    """
    if phase == "Lexical & Morphological":
        processed = series.apply(lexicon_clean)
        vec = vectorizer if vectorizer else CountVectorizer(binary=True, ngram_range=(1,2))
        X = vec.fit_transform(processed)
        return X, vec
    if phase == "Syntactic":
        processed = series.apply(pos_sequence)
        vec = vectorizer if vectorizer else TfidfVectorizer(max_features=5000)
        X = vec.fit_transform(processed)
        return X, vec
    if phase == "Semantic":
        df_feats = pd.DataFrame(series.apply(sentiment_feats).tolist(), columns=["polarity","subjectivity"])
        return df_feats, None
    if phase == "Discourse":
        processed = series.apply(discourse_summary)
        vec = vectorizer if vectorizer else CountVectorizer(ngram_range=(1,2), max_features=5000)
        X = vec.fit_transform(processed)
        return X, vec
    if phase == "Pragmatic":
        df_feats = pd.DataFrame(series.apply(pragmatic_counts).tolist(), columns=PRAGMATIC_TERMS)
        return df_feats, None
    return None, None

# ---------------------------
# 3) Model training & evaluation (kept logically the same)
# ---------------------------
def choose_classifier(name):
    if name == "Naive Bayes":
        return MultinomialNB()
    if name == "Decision Tree":
        return DecisionTreeClassifier(random_state=42, class_weight='balanced')
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, solver='liblinear', random_state=42, class_weight='balanced')
    if name == "SVM":
        return SVC(kernel='linear', C=0.5, random_state=42, class_weight='balanced')
    return None

def train_and_compare_models(df: pd.DataFrame, feature_phase: str) -> pd.DataFrame:
    # Binary mapping of labels
    REAL_LABELS = ["True", "No Flip", "Mostly True", "Half Flip", "Half True"]
    FAKE_LABELS = ["False", "Barely True", "Pants On Fire", "Full Flop"]

    def to_binary(lbl):
        if lbl in REAL_LABELS:
            return 1
        if lbl in FAKE_LABELS:
            return 0
        return np.nan

    df = df.copy()
    df['target_label'] = df['label'].apply(to_binary)
    df = df.dropna(subset=['target_label'])
    df = df[df['statement'].astype(str).str.len() > 10]

    X_raw = df['statement'].astype(str)
    y = df['target_label'].astype(int).values

    if len(np.unique(y)) < 2:
        st.error("Only one class present after mapping — cannot train.")
        return pd.DataFrame()

    # Feature extraction is now cached
    X_full, fitted_vec = extract_features_series(X_raw, feature_phase)
    if X_full is None:
        st.error("Feature extraction returned nothing.")
        return pd.DataFrame()

    if isinstance(X_full, pd.DataFrame):
        X_full = X_full.values

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    models = {
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear', random_state=42, class_weight='balanced'),
        "SVM": SVC(kernel='linear', C=0.5, random_state=42, class_weight='balanced')
    }

    metrics_accumulator = {m: [] for m in models.keys()}
    model_results = {}

    for name, model in models.items():
        st.caption(f"Training {name} — {N_SPLITS}-fold CV with SMOTE where applicable.")
        fold_metrics = {'accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'train_time': [], 'inference_time': []}

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_full, y)):
            # X_full is already transformed features (sparse matrix or numpy array)
            X_train = X_full[train_idx]
            X_test = X_full[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            start_t = time.time()
            try:
                if name == "Naive Bayes":
                    # Make sure features are non-negative counts/floats
                    # Sparse matrix support requires the matrix to be converted to array, only for non-sparse models
                    if hasattr(X_train, 'todense'):
                       X_train_final = np.abs(X_train).astype(float)
                    else:
                       X_train_final = np.abs(X_train)
                    clf = model
                    clf.fit(X_train_final, y_train)
                else:
                    pipe = ImbPipeline([('sampler', SMOTE(random_state=42, k_neighbors=3)), ('clf', model)])
                    pipe.fit(X_train, y_train)
                    clf = pipe

                train_time = time.time() - start_t
                start_inf = time.time()
                y_pred = clf.predict(X_test)
                inf_time = (time.time() - start_inf) * 1000.0

                fold_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
                fold_metrics['f1'].append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
                fold_metrics['precision'].append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
                fold_metrics['recall'].append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
                fold_metrics['train_time'].append(train_time)
                fold_metrics['inference_time'].append(inf_time)
            except Exception as e:
                st.warning(f"Fold {fold_idx+1} for {name} failed: {e}")
                for k in fold_metrics:
                    fold_metrics[k].append(0)
                continue

        if fold_metrics['accuracy']:
            model_results[name] = {
                "Model": name,
                "Accuracy": np.mean(fold_metrics['accuracy']) * 100,
                "F1-Score": np.mean(fold_metrics['f1']),
                "Precision": np.mean(fold_metrics['precision']),
                "Recall": np.mean(fold_metrics['recall']),
                "Training Time (s)": round(np.mean(fold_metrics['train_time']), 2),
                "Inference Latency (ms)": round(np.mean(fold_metrics['inference_time']), 2)
            }
        else:
            st.error(f"{name} failed for all folds.")
            model_results[name] = {"Model": name, "Accuracy": 0, "F1-Score": 0, "Precision": 0, "Recall": 0, "Training Time (s)": 0, "Inference Latency (ms)": 9999}

    return pd.DataFrame(list(model_results.values()))

# ---------------------------
# 4) Humour / critique (kept playful)
# ---------------------------
def pick_phase_roast(phase):
    roasts = {
        "Lexical & Morphological": ["Lexical: loud, proud, and word-counting."],
        "Syntactic": ["Syntactic: grammar police in action."],
        "Semantic": ["Semantic: feelings > facts."],
        "Discourse": ["Discourse: structure snobs unite."],
        "Pragmatic": ["Pragmatic: intent detectives on the case."]
    }
    return random.choice(roasts.get(phase, ["The models are speechless."]))

def pick_model_roast(model_name):
    roasts = {
        "Naive Bayes": ["Naive Bayes: small brain, big heart."],
        "Decision Tree": ["Decision Tree: asking questions until solved."],
        "Logistic Regression": ["Logistic Regression: boringly reliable."],
        "SVM": ["SVM: hard lines, dramatic splits."]
    }
    return random.choice(roasts.get(model_name, ["This model is having an existential crisis."]))

def witty_report(results_df: pd.DataFrame, phase: str) -> str:
    if results_df.empty:
        return "No trained models to roast — bring more data!"
    results_df['F1-Score'] = pd.to_numeric(results_df['F1-Score'], errors='coerce').fillna(0)
    best = results_df.loc[results_df['F1-Score'].idxmax()]
    best_model = best['Model']
    best_acc = best['Accuracy']
    best_f1 = best['F1-Score']
    return (
        f"**Top Model:** {best_model} — {best_acc:.2f}% accuracy, F1: {best_f1:.2f}\n\n"
        f"**Phase verdict:** {pick_phase_roast(phase)}\n\n"
        f"**Model personality:** {pick_model_roast(best_model)}"
    )

# ---------------------------
# 5) Cosine similarity utilities (new)
# ---------------------------
@st.cache_resource
def make_tfidf_vectorizer(corpus, max_features=10000):
    vec = TfidfVectorizer(max_features=max_features, stop_words='english')
    vec.fit(corpus)
    return vec

def statement_pair_similarity(tfidf_matrix, idx_a, idx_b):
    # tfidf_matrix: sparse matrix
    vecs = tfidf_matrix[[idx_a, idx_b]].toarray()
    sim = cosine_similarity(vecs[0].reshape(1, -1), vecs[1].reshape(1, -1))[0, 0]
    return float(sim)

def author_centroid_similarity(tfidf_matrix, authors_series: pd.Series, author_a, author_b):
    """
    Compute centroid (mean) TF-IDF vector per author, then cosine between the centroids.
    """
    # mask statements belonging to author
    mask_a = authors_series == author_a
    mask_b = authors_series == author_b
    if mask_a.sum() == 0 or mask_b.sum() == 0:
        return None
    
    # Ensure masking works correctly with sparse matrix indices
    if tfidf_matrix.ndim == 2:
      centroid_a = tfidf_matrix[mask_a.values].mean(axis=0)
      centroid_b = tfidf_matrix[mask_b.values].mean(axis=0)
    else:
      # Handle case where X_full is already a DataFrame or NumPy array (e.g., semantic/pragmatic features)
      centroid_a = tfidf_matrix[mask_a.values, :].mean(axis=0)
      centroid_b = tfidf_matrix[mask_b.values, :].mean(axis=0)
      
    # centroids might be matrix types — convert to np arrays
    centroid_a = np.asarray(centroid_a).ravel()
    centroid_b = np.asarray(centroid_b).ravel()
    
    # Handle single sample: reshape for similarity calculation
    if centroid_a.ndim == 1:
        centroid_a = centroid_a.reshape(1, -1)
    if centroid_b.ndim == 1:
        centroid_b = centroid_b.reshape(1, -1)

    sim = cosine_similarity(centroid_a, centroid_b)[0, 0]
    return float(sim)

def top_n_similar_pairs(tfidf_matrix, statements, top_n=3):
    """
    Compute pairwise cosine for all pairs and return top_n most similar pairs (excluding identical indices).
    """
    # For sparse matrices, convert to dense only when needed for full pairwise calculation
    if hasattr(tfidf_matrix, 'toarray'):
        dense = tfidf_matrix.toarray()
    else:
        dense = tfidf_matrix
        
    sim_mat = cosine_similarity(dense)
    n = sim_mat.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((i, j, sim_mat[i, j]))
    pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)
    top = pairs_sorted[:top_n]
    results = []
    for i, j, score in top:
        results.append({
            "idx_a": int(i),
            "idx_b": int(j),
            "score": float(score),
            "statement_a": statements.iloc[i],
            "statement_b": statements.iloc[j]
        })
    return results

# ---------------------------
# 6) Streamlit app layout
# ---------------------------
def app():
    st.set_page_config(page_title="VIOLET Fact-Check Console", layout="wide")
    
    # Custom, high-contrast header for a unique look
    st.markdown("""
    <style>
        .stApp {
            background-color: #0f172a; /* Deep Slate Background */
            color: #f8fafc; /* Light text */
        }
        .stTextInput > div > div > input, .stDateInput > label, .stSelectbox > label, .stSlider > label {
            color: #a855f7 !important; /* Violet accents on controls */
        }
        .stSelectbox div[role="listbox"] {
            background-color: #1e293b;
        }
        /* Custom box styling for sections */
        .custom-box {
            background-color: #1e293b; /* Darker Slate for sections */
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            border-left: 5px solid #a855f7; /* Violet accent bar */
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
        }
        /* Style for the buttons */
        .stButton button {
            background-color: #a855f7;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            border: none;
            padding: 10px 20px;
            transition: all 0.3s;
        }
        .stButton button:hover {
            background-color: #9333ea;
            box-shadow: 0 0 15px rgba(168, 85, 247, 0.6);
        }
        h2 {
            color: #a855f7; /* Violet subheaders */
            border-bottom: 2px solid #374151;
            padding-bottom: 10px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # New Header Look
    st.markdown("""
    <div style="
        padding: 30px;
        background: linear-gradient(145deg, #0f172a, #333333); 
        border-radius: 16px; 
        color: white;
        border: 2px solid #a855f7; 
        box-shadow: 0 4px 20px rgba(168, 85, 247, 0.4); /* Neon shadow */
    ">
        <h1 style="margin:0; font-size: 2.5em; font-weight: 800; color: #a855f7;">
            ⚡ VIOLET Fact-Check Console
        </h1>
        <p style="margin:0.2rem 0 0 0; opacity:0.8; font-size: 1.1em;">
            High-Velocity Linguistic Analysis and Model Comparison Engine.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # session_state initialization and initial file load
    if 'scraped_df' not in st.session_state:
        st.session_state['scraped_df'] = pd.DataFrame()
    if 'results_df' not in st.session_state:
        st.session_state['results_df'] = pd.DataFrame()
        
    # Attempt to load data from file on initial run
    if st.session_state['scraped_df'].empty and os.path.exists(SCRAPED_DATA_PATH):
        try:
            st.session_state['scraped_df'] = pd.read_csv(SCRAPED_DATA_PATH)
            st.info(f"Loaded {len(st.session_state['scraped_df'])} claims from local cache ('{SCRAPED_DATA_PATH}').")
        except Exception as e:
            st.warning(f"Could not load data from CSV: {e}")
            st.session_state['scraped_df'] = pd.DataFrame()


    col_left, col_mid, col_right = st.columns([1, 2, 2])

    # -------------------
    # LEFT: Data & Controls (Styled Section)
    # -------------------
    with col_left:
        st.markdown('<div class="custom-box">', unsafe_allow_html=True)
        st.header("1 — Data & Scrape Controls")
        min_date = pd.to_datetime('2007-01-01')
        max_date = pd.to_datetime('today').normalize()
        start_date = st.date_input("Start Date", min_value=min_date, max_value=max_date, value=pd.to_datetime('2023-01-01'))
        end_date = st.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)

        if st.button("Scrape Politifact"):
            if start_date > end_date:
                st.error("Start can't be after End.")
            else:
                with st.spinner("Scraping..."):
                    df = collect_claims_within_range(pd.to_datetime(start_date), pd.to_datetime(end_date))
                if not df.empty:
                    st.session_state['scraped_df'] = df
                    st.success(f"Collected {len(df)} claims.")
                else:
                    st.warning("No claims collected. Try a different date range.")

        st.divider()
        st.header("2 — Modeling Options")
        phases = ["Lexical & Morphological", "Syntactic", "Semantic", "Discourse", "Pragmatic"]
        selected_phase = st.selectbox("Feature phase:", phases, index=0)

        if st.button("Run Model Showdown"):
            if st.session_state['scraped_df'].empty:
                st.error("Scrape data first.")
            else:
                with st.spinner("Training models..."):
                    # Clear caches related to the old data/features before running new training
                    extract_features_series.clear()
                    # train_and_compare_models.clear() # Avoid clearing if function itself isn't cached
                    
                    results = train_and_compare_models(st.session_state['scraped_df'], selected_phase)
                    st.session_state['results_df'] = results
                    st.session_state['selected_phase_run'] = selected_phase
                st.success("Model training complete.")
        st.markdown('</div>', unsafe_allow_html=True)


    # -------------------
    # MIDDLE: Metrics & Visuals (Styled Section)
    # -------------------
    with col_mid:
        st.markdown('<div class="custom-box">', unsafe_allow_html=True)
        st.header("3 — Performance Dashboard")
        if st.session_state['results_df'].empty:
            st.info("No results yet. Run training to populate metrics.")
        else:
            results_df = st.session_state['results_df']
            st.subheader(f"Results — {st.session_state.get('selected_phase_run','')}")
            st.dataframe(results_df[['Model','Accuracy','F1-Score','Training Time (s)','Inference Latency (ms)']], height=220, use_container_width=True)
            
            st.divider()
            st.subheader("Metric Plot")
            metrics = ['Accuracy','F1-Score','Precision','Recall','Training Time (s)','Inference Latency (ms)']
            chosen_metric = st.selectbox("Metric to display:", metrics, index=1, key='metric_select')
            
            # Using st.bar_chart (default Streamlit)
            df_plot = results_df[['Model', chosen_metric]].set_index('Model')
            st.bar_chart(df_plot, color="#a855f7") # Force violet color
            

            st.divider()
            st.subheader("Model Scatter (Speed vs Quality)")
            x_axis = st.selectbox("X axis:", ['Training Time (s)', 'Inference Latency (ms)'], index=1, key='scatter_x')
            y_axis = st.selectbox("Y axis:", ['Accuracy','F1-Score','Precision','Recall'], index=0, key='scatter_y')
            
            # Custom Matplotlib Plotting for unique graph styling
            fig, ax = plt.subplots(figsize=(7,5), facecolor='#1e293b') # Darker Slate background for the plot area
            ax.scatter(results_df[x_axis], results_df[y_axis], s=200, alpha=0.9, color='#a855f7', edgecolors='white', linewidths=1.5) # Violet scatter
            
            for _, r in results_df.iterrows():
                # Annotate with white/light color
                ax.annotate(r['Model'], (r[x_axis] + 0.01*results_df[x_axis].max(), r[y_axis]), fontsize=10, color='white')
            
            # Set axis, title, and tick colors to match the dark theme
            ax.set_xlabel(x_axis, color='#d1d5db')
            ax.set_ylabel(y_axis, color='#d1d5db')
            ax.set_title(f"{y_axis} vs {x_axis}", color='white')
            
            ax.tick_params(axis='x', colors='#9ca3af')
            ax.tick_params(axis='y', colors='#9ca3af')
            
            # Set grid and spine colors
            for spine in ax.spines.values():
                spine.set_color('#374151')
            
            ax.grid(True, linestyle='-', alpha=0.3, color='#374151') # Dark grid lines
            fig.patch.set_facecolor('#0f172a') # Background outside the plot (Deep Slate)
            st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)


    # -------------------
    # RIGHT: Critique + Cosine similarity explorer (Styled Section)
    # -------------------
    with col_right:
        st.markdown('<div class="custom-box">', unsafe_allow_html=True)
        st.header("4 — Critique & Similarity Explorer")

        if st.session_state['results_df'].empty:
            st.info("Model critiques will appear here after running training.")
        else:
            critique = witty_report(st.session_state['results_df'], st.session_state.get('selected_phase_run',''))
            st.markdown(critique)

        st.divider()
        st.subheader("Statement Similarity Explorer (TF-IDF + Cosine)")

        df_src = st.session_state['scraped_df']
        if df_src.empty:
            st.info("Load a scraped dataset first to use the similarity explorer.")
        else:
            # Create a local DataFrame with a clean 0-based index for easy access
            df_sim = df_src.reset_index(drop=True).copy()
            
            # Prepare TF-IDF on statements (cache vectorizer)
            corpus = df_sim['statement'].astype(str).tolist()
            # make_tfidf_vectorizer is already cached
            tfidf_vect = make_tfidf_vectorizer(corpus) 
            tfidf_mat = tfidf_vect.transform(corpus)

            # Option mode: statement-pair or author-pair
            sim_mode = st.radio("Similarity Mode:", ["Statement vs Statement", "Author vs Author", "Top similar pairs across dataset"], index=0)

            if sim_mode == "Statement vs Statement":
                st.write("Pick two statements from the dataset to compare. Indices are 0-based.")
                
                # Use df_sim for index selection, which is 0-based
                idx_a = st.selectbox("Statement A (index)", options=list(df_sim.index), format_func=lambda i: f"{i}: {df_sim.loc[i,'statement'][:80]}...", key='s_idx_a')
                idx_b = st.selectbox("Statement B (index)", options=list(df_sim.index), format_func=lambda i: f"{i}: {df_sim.loc[i,'statement'][:80]}...", key='s_idx_b')
                
                if st.button("Compute Statement Similarity"):
                    if idx_a == idx_b:
                        st.warning("You selected the same statement twice. Similarity = 1.0 by definition.")
                    # idx_a and idx_b are the correct 0-based row indices for tfidf_mat
                    sim_score = statement_pair_similarity(tfidf_mat, idx_a, idx_b)
                    st.metric("Cosine similarity", f"{sim_score:.4f}")
                    st.markdown("**Statement A:**"); st.write(df_sim.loc[idx_a, 'statement'])
                    st.markdown("**Statement B:**"); st.write(df_sim.loc[idx_b, 'statement'])

            elif sim_mode == "Author vs Author":
                st.write("Compare the average statement style of two authors (TF-IDF centroids).")
                authors = sorted(df_sim['author'].fillna("Unknown").unique().tolist())
                author_a = st.selectbox("Author A", authors, key='author_a')
                author_b = st.selectbox("Author B", authors, key='author_b')
                if st.button("Compute Author-Centroid Similarity"):
                    if author_a == author_b:
                        st.warning("Same author selected — similarity = 1.0.")
                    # Use df_sim for author lookup
                    sim = author_centroid_similarity(tfidf_mat, df_sim['author'].fillna("Unknown"), author_a, author_b)
                    if sim is None:
                        st.error("One of the chosen authors has no statements in the dataset.")
                    else:
                        st.metric("Author centroid cosine similarity", f"{sim:.4f}")
                        st.info(f"Author A ({author_a}) — {df_sim[df_sim['author'].fillna('Unknown')==author_a].shape[0]} statements")
                        st.info(f"Author B ({author_b}) — {df_sim[df_sim['author'].fillna('Unknown')==author_b].shape[0]} statements")

            else:
                st.write("Top similar statement pairs in the dataset (TF-IDF cosine). Indices are 0-based.")
                top_k = st.slider("Top K pairs", min_value=1, max_value=10, value=3)
                if st.button("Find Top Similar Pairs"):
                    top_pairs = top_n_similar_pairs(tfidf_mat, df_sim['statement'], top_n=top_k)
                    if not top_pairs:
                        st.warning("No pairs found (dataset may be too small).")
                    else:
                        for i, pair in enumerate(top_pairs, start=1):
                            st.markdown(f"**#{i} — Score: {pair['score']:.4f}**")
                            # Use df_sim for statement lookup by 0-based index
                            st.markdown(f"- **A (idx {pair['idx_a']}):** {df_sim.loc[pair['idx_a'], 'statement']}")
                            st.markdown(f"- **B (idx {pair['idx_b']}):** {df_sim.loc[pair['idx_b'], 'statement']}")
                            st.divider()
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    app()
