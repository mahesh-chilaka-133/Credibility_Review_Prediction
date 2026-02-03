import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from textblob import TextBlob
import textstat
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# 1. Page Configuration & Styling
# ==========================================
st.set_page_config(
    page_title="ReviewTrust",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1.1 Hide Streamlit Standard UI Elements (Academic Requirement) ---
hide_streamlit_style = """
<style>
/* Remove the "Deploy" button */
.stDeployButton {
    display: none !important;
}



/* Remove the "Running..." status widget */
[data-testid="stStatusWidget"] {
    display: none !important;
}

/* Ensure the header and toolbar are VISIBLE so the sidebar toggle can be seen */
header, [data-testid="stToolbar"] {
    visibility: visible !important;
    background: transparent !important;
    height: auto !important;
}

/* Hide footer */
footer {
    display: none !important;
}

/* Force sidebar toggle to be visible & colored correctly */
[data-testid="stSidebarCollapsedControl"] {
    display: block !important;
    visibility: visible !important;
    color: #1f3fae !important;
}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- 1.2 Custom Academic CSS Theme ---
st.markdown("""
    <style>
    /* Global Font & Background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #64748B; /* Text Secondary */
    }
    
    /* Main App Background */
    .stApp {
        background-color: #FAFAFA;
        margin-top: 0px; /* Remove margin */
    }
    
    /* Reduce top padding for main content and sidebar */
    .block-container {
        padding-top: 1rem !important; /* Bring content up */
        padding-bottom: 1rem !important;
    }
    
    /* Hide the default Streamlit header decoration if causing gap */
    header[data-testid="stHeader"] {
        background-color: transparent;
    }
    
    /* Sidebar Background - LIGHT PURPLE BRAND */
    [data-testid="stSidebar"] {
        background-color: #F8F5FF; /* Light Lavender */
        border-right: 1px solid #E9D5ED;
    }
    
    /* Sidebar Text Adjustments (Dark Purple on Light) */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] div, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span, [data-testid="stSidebar"] p {
        color: #68217A !important; /* Brand Purple Text */
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        color: #0F172A; /* Text Primary */
    }

    h1 {
        font-weight: 700;
        font-size: 2.2rem;
        letter-spacing: -0.02em;
        color: #68217A; /* Brand Purple */
    }
    h2 {
        color: #009E49; /* Brand Green */
        font-weight: 600;
        font-size: 1.5rem;
        margin-top: 1.5rem;
        letter-spacing: -0.01em;
    }
    h3 {
        color: #64748B; /* Text Secondary */
        font-weight: 500;
        font-size: 1.125rem;
    }
    
    p, div, label, li, span {
        color: #64748B;
        line-height: 1.6;
    }
    
    /* Ensure sidebar toggle button is visible */
    [data-testid="stSidebarCollapsedControl"] {
        z-index: 999999;
        color: #334155;
        background-color: #F1F5F9;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 4px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    /* Add padding to top of main container */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem;
    }
    
    /* Cards - Minimal Theme */
    /* Cards - Minimal Theme */
    .research-card, .workflow-step {
        background-color: #FFFFFF; /* Card Background */
        border-radius: 8px;
        padding: 24px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px -1px rgba(0, 0, 0, 0.1); 
        border: 1px solid #E2E8F0;
        margin-bottom: 20px;
        min-height: 280px; 
        transition: box-shadow 0.2s ease-in-out;
    }
    .metric-container {
        background-color: #FFFFFF;
        border-radius: 8px;
        padding: 16px; /* Reduced Padding */
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        border: 1px solid #E2E8F0;
        margin-bottom: 12px;
        min-height: 100px; /* Much smaller height */
        text-align: center;
        transition: all 0.2s ease;
    }
    .research-card:hover, .workflow-step:hover, .metric-container:hover {
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border-color: #CBD5E1; 
    }
    
    /* Workflow Steps specific */
    .workflow-step {
        text-align: center;
        height: 100%;
        min-height: 240px;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        align-items: center;
        border-top: 4px solid #009E49; /* Brand Green */
        padding-top: 32px;
    }
    .step-icon {
        font-size: 28px;
        margin-bottom: 15px;
        background: #F0FDF4; /* Light Green Tint */
        color: #009E49;
        padding: 12px;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        display: flex;
        justify-content: center;
        align-items: center;
        border: 1px solid #E2E8F0;
    }
    .step-title {
        font-weight: 600;
        color: #0F172A;
        font-size: 1rem;
    }

    /* Result Indicators */
    .result-box-container {
        background-color: #FFFFFF;
        padding: 30px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        margin-top: 25px;
        border: 1px solid #E2E8F0;
    }
    .credible-result {
        background: linear-gradient(to bottom, #FFFFFF, #F0FDF4);
        border-top: 5px solid #10B981; /* Green */
    }
    .fake-result {
        background: linear-gradient(to bottom, #FFFFFF, #FEF2F2);
        border-top: 5px solid #EF4444; /* Red */
    }
    .result-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 8px;
        letter-spacing: -0.01em;
        color: #0F172A;
    }
    .result-score {
        font-size: 3.5rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -2px;
        line-height: 1.2;
    }
    
    /* Feature Metrics */
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #334155;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #64748B;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    /* Table Styling */
    thead tr th:first-child {display:none}
    tbody th {display:none}
    
    /* Buttons - BRAND GREEN */
    div.stButton > button:first-child {
        background-color: #009E49; /* Brand Green */
        color: #FFFFFF !important;
        border: none;
        padding: 12px 24px;
        border-radius: 6px;
        font-size: 15px;
        font-weight: 600;
        letter-spacing: 0.3px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        transition: background-color 0.2s ease;
        text-transform: none;
    }
    div.stButton > button:first-child:hover {
        background-color: #007A39; /* Darker Green */
        color: #FFFFFF !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: none;
    }
    div.stButton > button p {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    /* Custom Navigation Styling (Radio Group Transformation) */
    [data-testid="stSidebar"] [data-testid="stRadio"] > label {
        display: none !important; /* Hide the "Navigation" label */
    }
    
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label {
        background-color: transparent;
        padding: 8px 12px; /* Reduced Padding */
        margin-bottom: 2px;
        border-radius: 6px;
        transition: all 0.2s ease;
        border: 1px solid transparent;
        color: #68217A !important; /* Brand Purple Text */
        font-size: 0.9rem; /* Slightly smaller font */
    }
    
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        background-color: #E9D5ED; /* Darker Lavender Hover */
        color: #4A148C !important; /* Deep Purple Text on Hover */
        transform: translateX(4px); /* Slght nudging effect */
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    /* Radio circle hiding */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label > div:first-child {
        display: none;
    }
    
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label > div:nth-child(2) {
        margin-left: 0 !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label[data-baseweb="radio"] {
        background-color: transparent !important;
    }
       
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Resource Loading
# ==========================================
@st.cache_resource
def load_resources():
    try:
        model = load_model('model.h5')
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading system resources: {e}")
        return None, None

model, scaler = load_resources()

# NLTK Init
for r in ['stopwords', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(f'corpora/{r}')
    except LookupError:
        nltk.download(r, quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ==========================================
# 3. Logic & Helpers
# ==========================================
def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

def extract_features(text, stars):
    cleaned = preprocess_text(text)
    
    feat = {}
    feat['review_length'] = len(text)
    feat['word_count'] = len(text.split())
    
    blob = TextBlob(text)
    feat['polarity'] = blob.sentiment.polarity
    feat['subjectivity'] = blob.sentiment.subjectivity
    
    feat['extremity'] = 1 if stars in [1, 5] else 0
    feat['dataset_avg'] = 3.75
    feat['external_consistency'] = abs(stars - feat['dataset_avg'])
    
    normalized_stars = (stars - 3) / 2
    feat['internal_consistency'] = abs(feat['polarity'] - normalized_stars)
    
    feat['readability'] = textstat.flesch_reading_ease(text) if len(text) > 20 else 50
    
    # Feature description dictionary for tooltips
    feature_tooltips = {
        'review_length': "Total characters in the review.",
        'word_count': "Total number of words.",
        'polarity': "Sentiment score (-1 to +1).",
        'subjectivity': "Subjectivity score (0 to 1).",
        'readability': "Flesch Reading Ease score.",
        'extremity': "Binary flag (1 for 1 or 5 stars).",
        'external_consistency': "Deviation from product avg rating.",
        'internal_consistency': "Alignment between text & rating."
    }
    
    feature_vector = np.array([[
        feat['review_length'], feat['word_count'], feat['polarity'], feat['subjectivity'],
        feat['readability'], feat['extremity'], feat['external_consistency'], feat['internal_consistency']
    ]])
    
    return feature_vector, feat, feature_tooltips

# ==========================================
# 4. Page Routing
# ==========================================
if 'page' not in st.session_state:
    st.session_state.page = "Home"

# Logic for closing sidebar on mobile after selection
if st.session_state.get('nav_clicked', False):
    st.session_state.nav_clicked = False
    st.components.v1.html(
        """
        <script>
            const sidebar = window.parent.document.querySelector('section[data-testid="stSidebar"]');
            if (sidebar && window.parent.innerWidth < 992) {
                // Find the collapse button and click it
                const closeButton = window.parent.document.querySelector('[data-testid="stSidebarCollapseButton"]');
                if (closeButton) {
                    closeButton.click();
                }
            }
        </script>
        """,
        height=0,
        width=0,
    )

# Sidebar
with st.sidebar:
    st.title("⚖️ ReviewTrust")
    st.caption("Deep Learning Framework")
    st.markdown("---")
    
    selection = st.radio(
        "Navigation",
        ["Home", "Review Analyzer", "Model Information", "About Research"],
        index=["Home", "Review Analyzer", "Model Information", "About Research"].index(st.session_state.page),
        label_visibility="collapsed" # Hide the label explicitly
    )
    
    if selection != st.session_state.page:
        st.session_state.page = selection
        st.session_state.nav_clicked = True # Trigger sidear close on next run
        st.rerun()
        
    st.markdown("---")
    st.info("💡 **Detection Engine**\nMLP Neural Network\naccuracy: 95.19%")

# ==========================================
# 5. Page Content
# ==========================================

# --- HOME PAGE ---
if st.session_state.page == "Home":
    st.markdown("""
        <div style="text-align: center; padding: 40px 0;">
            <h1 style="margin-bottom: 12px;">ReviewTrust</h1>
            <h3 style="color: #9CA3AF; font-weight: 400; max-width: 800px; margin: 0 auto;">
                A Deep Learning Approach to Online Review Trustworthiness
            </h3>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 🔄 System Architecture Plan")
    st.markdown("---")
    
    # Workflow
    steps = [
        ("📝 Review Input", "User submits text & star rating"),
        ("🧹 Preprocessing", "Stop-words removal & Lemmatization"),
        ("⚙️ Feature Extraction", "Behavioral & Linguistic Analysis"),
        ("🧠 Neural Inference", "MLP Model Processing (8 Layers)"),
        ("✅ Credibility Score", "Trust Classification Output")
    ]
    
    cols = st.columns(len(steps))
    for i, (title, desc) in enumerate(steps):
        with cols[i]:
            st.markdown(f"""
            <div class="workflow-step">
                <div class="step-icon">{'📋' if i==0 else '🔧' if i==1 else '📊' if i==2 else '🤖' if i==3 else '🛡️'}</div>
                <div class="step-title">{title}</div>
                <div style="font-size:13px; color:#57606a; margin-top:8px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.markdown("## ❓ Motivation & Impact")
        st.markdown("""
        <div class="research-card">
            <p><strong>Trust is the currency of the digital economy.</strong> Deceptive reviews (opinion spam) undermine this trust.</p>
            <ul>
                <li><strong>Privacy-First:</strong> Our model analyzes the <em>what</em> (content), not the <em>who</em> (user history).</li>
                <li><strong>Consistency Checking:</strong> We detect subtle contradictions between sentiment and rating.</li>
                <li><strong>Scalability:</strong> Designed for high-throughput e-commerce platforms.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
            
    with c2:
        st.markdown("## 🎯 Core Objectives")
        st.markdown("""
        <div class="research-card">
            <ol style="padding-left: 20px;">
                <li>Detect deceptive opinion spam.</li>
                <li>Preserve user privacy.</li>
                <li>Benchmark Deep Learning vs Classical ML.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Start Analysis 🚀", use_container_width=True):
        st.session_state.page = "Review Analyzer"
        st.rerun()

# --- ANALYZER PAGE ---
elif st.session_state.page == "Review Analyzer":
    st.title("🔍 Review Authenticity Analyzer")
    st.markdown("Input the review content below to generate a credibility profile.")
    
    # Input
    with st.container():

        c1, c2 = st.columns([2.5, 1])
        with c1:
            review_text = st.text_area("Review Text", height=180, placeholder="Paste user review content here...")
        with c2:
            star_rating = st.slider("Star Rating", 1, 5, 5)
            st.caption("Associated numerical rating")


    if st.button("Analyze Credibility", type="primary"):
        if not review_text.strip():
            st.warning("⚠️ Input is empty.")
        elif model is None:
            st.error("🚨 Model not connected.")
        else:
            with st.spinner("Running inference..."):
                vec, feats, tooltips = extract_features(review_text, star_rating)
                vec_scaled = scaler.transform(vec)
                prob = model.predict(vec_scaled)[0][0]
                
                is_credible = prob > 0.5
                confidence = prob if is_credible else 1 - prob
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Result Block
                r_col, v_col = st.columns([1, 1.2])
                
                with r_col:
                    res_class = "credible-result" if is_credible else "fake-result"
                    res_title = "✅ CREDIBLE" if is_credible else "⚠️ NON-CREDIBLE"
                    res_color = "#2ecc71" if is_credible else "#e74c3c"
                    
                    st.markdown(f"""
                    <div class="result-box-container {res_class}">
                        <div class="result-title" style="color: {res_color}">{res_title}</div>
                        <div class="result-score">{confidence:.1%}</div>
                        <div style="color: #7f8c8d; font-weight:600; font-size:14px;">CONFIDENCE SCORE</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with v_col:
                    # Radar Chart
                    metrics = {
                        'Subjectivity': feats['subjectivity'],
                        'Inconsistency': min(feats['internal_consistency'], 1.0),
                        'Sentiment': (feats['polarity'] + 1) / 2, 
                        'Readability': min(feats['readability']/100, 1)
                    }
                    
                    fig = px.line_polar(r=list(metrics.values()), theta=list(metrics.keys()), 
                                      line_close=True, range_r=[0,1],
                                      color_discrete_sequence=['#68217A'])
                    fig.update_traces(fill='toself')
                    fig.update_layout(margin=dict(t=10, b=10, l=40, r=40), height=220, 
                                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)

                # Feature Grid
                st.markdown("### 📊 Linguistic Profiles")
                f1, f2, f3, f4 = st.columns(4)
                
                def f_card(label, val, key):
                    return f"""
                    <div class="metric-container" title="{tooltips[key]}">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{val}</div>
                    </div>"""
                
                with f1: st.markdown(f_card("Word Count", feats['word_count'], 'word_count'), unsafe_allow_html=True)
                with f2: st.markdown(f_card("Readability", f"{feats['readability']:.0f}", 'readability'), unsafe_allow_html=True)
                with f3: st.markdown(f_card("Subjectivity", f"{feats['subjectivity']:.2f}", 'subjectivity'), unsafe_allow_html=True)
                with f4: st.markdown(f_card("Consistency", f"{feats['internal_consistency']:.2f}", 'internal_consistency'), unsafe_allow_html=True)

# --- MODEL INFO PAGE ---
elif st.session_state.page == "Model Information":
    st.title("🧠 Neural Network Architecture")
    
    c1, c2 = st.columns([1.5, 1])
    
    with c1:
        st.markdown("### 🏆 Performance Analysis")
        # Final DataFrame
        df_res = pd.DataFrame({
            "Classifier": ["MLP (Proposed)", "XGBoost", "Random Forest", "SVM", "Logistic Regression"],
            "Accuracy": ["95.19%", "92.2%", "92.1%", "91.6%", "88.2%"],
            "AUC Score": ["0.972", "0.899", "0.900", "0.901", "0.899"]
        })
        st.table(df_res)
        
        st.success("The **MLP (Multi-Layer Perceptron)** uses 3 hidden dense layers with Dropout regularization (0.3) to prevent overfitting, achieving state-of-the-art accuracy.")

    with c2:
        st.markdown("### ⚙️ MLP Configuration")
        st.markdown("""
        <div class="research-card">
        <ul style="list-style-type: none; padding-left: 0;">
            <li style="margin-bottom:8px">🔹 <strong>Input:</strong> 8 Scaled Features</li>
            <li style="margin-bottom:8px">⬇</li>
            <li style="margin-bottom:8px">🔹 <strong>Dense 1:</strong> 64 Units (ReLU)</li>
            <li style="margin-bottom:8px">⬇ 🛑 Dropout (0.3)</li>
            <li style="margin-bottom:8px">🔹 <strong>Dense 2:</strong> 32 Units (ReLU)</li>
            <li style="margin-bottom:8px">⬇ 🛑 Dropout (0.2)</li>
            <li style="margin-bottom:8px">🔹 <strong>Dense 3:</strong> 16 Units (ReLU)</li>
            <li style="margin-bottom:8px">⬇</li>
            <li>🔹 <strong>Output:</strong> Sigmoid (Probabilistic)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# --- ABOUT PAGE ---
elif st.session_state.page == "About Research":
    st.title("📚 Research Context")
    
    st.markdown("### 📄 Abstract")
    st.markdown("""
    <div class="research-card" style="text-align: justify;">
        Customer purchase choices are strongly shaped by online reviews; however, deceptive or manipulated reviews reduce their reliability for both consumers and businesses. 
        To address this issue, we developed a predictive framework that determines the credibility of a review based on textual content and consistency-related checks, 
        without depending on reviewer specific details that could raise privacy concerns. Our methodology incorporates six key features that influence credibility: 
        review length, subjectivity, readability, extremity, internal consistency, and external consistency. Attributes extracted from Yelp reviews were analyzed 
        through both traditional ML algorithms and a neural network model were employed for analysis. Among the tested approaches, the Multi-Layer Perceptron (MLP) 
        achieved the best performance, reporting 95.19% accuracy and an AUC of 0.972, surpassing the classical models. SVM and XGBoost also delivered strong results, 
        while Random Forest and KNN showed competitive performance. Feature importance analysis indicated that subjectivity, review length, and rating consistency 
        exert the strongest influence on credibility prediction. These results highlight that integrating linguistic features with consistency based indicators, 
        alongside advanced classifiers, can provide an effective and scalable solution for detecting fake reviews. This framework enhances the reliability of 
        online reviews, thereby assisting consumers in decision-making and improving confidence in digital platforms.
    </div>
    """, unsafe_allow_html=True)
    
    # Use tabs for cleaner layout on mobile
    tab1, tab2 = st.tabs(["💾 Dataset Details", "🔮 Future Work"])
    
    with tab1:
        st.markdown("""
        <div class="research-card">
            <h4>Yelp Academic Dataset</h4>
            <p>The model was trained on a rigorous subset of the Yelp Academic Dataset, focusing on restaurant and product reviews.</p>
            <ul>
                <li><strong>Total Samples:</strong> ~60,000 processed reviews</li>
                <li><strong>Class Balance:</strong> 50% Deceptive (Generated/Filtered) vs 50% Truthful</li>
                <li><strong>Preprocessing:</strong> NLP pipeline including noise removal, lemmatization, and tokenization.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with tab2:
        st.markdown("""
        <div class="research-card">
            <h4>Roadmap & Enhancements</h4>
            <p>Current limitations are addressed in the proposed future scope:</p>
            <ul>
                <li><strong>Transformer Integration:</strong> Incorporating BERT/RoBERTa embeddings for deeper semantic understanding beyond statistical features.</li>
                <li><strong>Real-Time API:</strong> Developing a browser extension for real-time analysis on Amazon/Flipkart.</li>
                <li><strong>Multilingual Support:</strong> extending the linguistic feature set to support non-English reviews.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    st.caption("ReviewTrust | Standalone Research Demonstration | v1.0")
