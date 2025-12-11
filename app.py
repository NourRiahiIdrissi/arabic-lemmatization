import streamlit as st
import json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import numpy as np
from pathlib import Path

# Buckwalter to Arabic conversion
def buckwalter_to_arabic(text):
    """Convert Buckwalter transliteration to Arabic script"""
    buckwalter_map = {
        "'": "ء", ">": "أ", "&": "ؤ", "<": "إ", "}": "ئ",
        "A": "ا", "b": "ب", "p": "ة", "t": "ت", "v": "ث",
        "j": "ج", "H": "ح", "x": "خ", "d": "د", "*": "ذ",
        "r": "ر", "z": "ز", "s": "س", "$": "ش", "S": "ص",
        "D": "ض", "T": "ط", "Z": "ظ", "E": "ع", "g": "غ",
        "_": "ـ", "f": "ف", "q": "ق", "k": "ك", "l": "ل",
        "m": "م", "n": "ن", "h": "ه", "w": "و", "Y": "ى",
        "y": "ي", "F": "ً", "N": "ٌ", "K": "ٍ", "a": "َ",
        "u": "ُ", "i": "ِ", "~": "ّ", "o": "ْ", "`": "ٰ",
        "{": "ٱ", "|": "آ"
    }
    result = ""
    for char in text:
        result += buckwalter_map.get(char, char)
    return result

# Arabic to Buckwalter conversion
def arabic_to_buckwalter(text):
    """Convert Arabic script to Buckwalter transliteration"""
    arabic_map = {
        "ء": "'", "أ": ">", "ؤ": "&", "إ": "<", "ئ": "}",
        "ا": "A", "ب": "b", "ة": "p", "ت": "t", "ث": "v",
        "ج": "j", "ح": "H", "خ": "x", "د": "d", "ذ": "*",
        "ر": "r", "ز": "z", "س": "s", "ش": "$", "ص": "S",
        "ض": "D", "ط": "T", "ظ": "Z", "ع": "E", "غ": "g",
        "ـ": "_", "ف": "f", "ق": "q", "ك": "k", "ل": "l",
        "م": "m", "ن": "n", "ه": "h", "و": "w", "ى": "Y",
        "ي": "y", "ً": "F", "ٌ": "N", "ٍ": "K", "َ": "a",
        "ُ": "u", "ِ": "i", "ّ": "~", "ْ": "o", "ٰ": "`",
        "ٱ": "{", "آ": "|"
    }
    result = ""
    for char in text:
        result += arabic_map.get(char, char)
    return result

# Page configuration
st.set_page_config(
    page_title="Arabic Lemmatization System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Professional styling
st.markdown("""
    <style>
    /* Main header */
    .main-header {
        font-size: 2.2rem;
        font-weight: 600;
        color: #1a1a1a;
        text-align: center;
        margin-bottom: 1rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .subtitle {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Remove default Streamlit padding */
    .block-container {
        padding-top: 2rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Metrics */
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
    }
    
    /* Tables */
    .dataframe {
        font-size: 0.9rem;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background-color: #0066cc;
        color: white;
        font-weight: 500;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
    }
    
    .stButton>button:hover {
        background-color: #0052a3;
    }
    
    /* Download button */
    .stDownloadButton>button {
        background-color: #28a745;
        color: white;
        border: none;
    }
    
    .stDownloadButton>button:hover {
        background-color: #218838;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer(model_choice, hf_username):
    """Load the trained model and tokenizer from Hugging Face"""
    try:
        # Map model choice to Hugging Face repo
        if model_choice == "AraBERT v02":
            repo_id = f"{hf_username}/arabertv02-lemmatization"
        else:
            repo_id = f"{hf_username}/camelbert-lemmatization"
        
        st.info(f"Loading model from Hugging Face: {repo_id}")
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        model = AutoModelForTokenClassification.from_pretrained(repo_id)
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

@st.cache_data
def load_label_map(label_path):
    """Load label2id mapping"""
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            label2id = json.load(f)
        id2label = {v: k for k, v in label2id.items()}
        return label2id, id2label
    except Exception as e:
        st.error(f"Error loading label map: {str(e)}")
        return None, None

@st.cache_data
def load_results(results_path):
    """Load model comparison results"""
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Could not load results: {str(e)}")
        return None

def predict_lemmas(text, tokenizer, model, id2label):
    """Predict lemmas for input text"""
    # Split text into tokens
    original_tokens = text.strip().split()
    
    # Convert Arabic tokens to Buckwalter for the model
    tokens = [arabic_to_buckwalter(token) for token in original_tokens]
    
    if not tokens:
        return [], []
    
    # Tokenize
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    
    # Predict
    with torch.no_grad():
        outputs = model(**encoding)
        predictions = torch.argmax(outputs.logits, dim=-1)
    
    # Align predictions with original tokens
    word_ids = encoding.word_ids(batch_index=0)
    predicted_lemmas = []
    
    prev_word_id = None
    for idx, word_id in enumerate(word_ids):
        if word_id is not None and word_id != prev_word_id:
            label_id = predictions[0][idx].item()
            lemma = id2label.get(label_id, "UNK")
            # Convert from Buckwalter to Arabic
            lemma = buckwalter_to_arabic(lemma)
            predicted_lemmas.append(lemma)
        prev_word_id = word_id
    
    return original_tokens, predicted_lemmas

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">Arabic Lemmatization System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">A machine learning system for Arabic morphological analysis</p>', unsafe_allow_html=True)
    
    # Sidebar Configuration
    st.sidebar.header("Model Configuration")
    
    # Model selection
    #available_models = []
    #if Path("final_bert-base-arabertv02").exists():
     #   available_models.append("AraBERT v02")
    #if Path("final_bert-base-arabic-camelbert-msa").exists():
     #   available_models.append("CAMeL-BERT MSA")
    
    if not available_models:
        st.error("No trained models found. Please ensure model directories exist.")
        st.stop()
    # Model selection
    available_models = ["AraBERT v02", "CAMeL-BERT MSA"]
    
    model_choice = st.sidebar.selectbox(
        "Select Model",
        available_models,
        help="Choose which trained model to use for prediction"
    )
    
    # Map selection to path
    model_path_map = {
        "AraBERT v02": "final_bert-base-arabertv02",
        "CAMeL-BERT MSA": "final_bert-base-arabic-camelbert-msa"
    }
    model_path = model_path_map[model_choice]
    
    # Load resources
    with st.spinner("Loading model and resources..."):
        # Add your Hugging Face username here
        HF_USERNAME = "nourriahiidrissi"  # Replace with your actual username
        tokenizer, model = load_model_and_tokenizer(model_choice, HF_USERNAME)
        label2id, id2label = load_label_map("merged_label2id.json")
        results = load_results("results.json")
    
    if tokenizer is None or model is None or label2id is None:
        st.error("Failed to load required resources. Please check your file paths.")
        st.stop()
    
    st.sidebar.success(f"Model loaded: {model_choice}")
    st.sidebar.info(f"Total labels: {len(label2id)}")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Lemmatization", "Model Performance", "Error Analysis", "About"])
    
    # Tab 1: Lemmatization Interface
    with tab1:
        st.header("Lemmatization Interface")
        st.write("Enter Arabic text to generate lemmas for each token.")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Input options
            input_method = st.radio(
                "Input Method",
                ["Text Input", "Example Sentences"],
                horizontal=True
            )
            
            if input_method == "Text Input":
                user_input = st.text_area(
                    "Enter Arabic text (space-separated tokens):",
                    height=120,
                    placeholder="بسم الله الرحمن الرحيم",
                    help="Enter Arabic text with tokens separated by spaces"
                )
            else:
                examples = [
                    "بسم الله الرحمن الرحيم",
                    "الحمد لله رب العالمين",
                    "قل هو الله أحد",
                    "إنا أعطيناك الكوثر"
                ]
                user_input = st.selectbox("Select an example:", examples)
        
        with col2:
            st.markdown("### Statistics")
            if user_input:
                token_count = len(user_input.strip().split())
                st.metric("Tokens", token_count)
                st.metric("Characters", len(user_input))
        
        if st.button("Generate Lemmas", type="primary", use_container_width=True):
            if user_input.strip():
                with st.spinner("Processing..."):
                    tokens, lemmas = predict_lemmas(user_input, tokenizer, model, id2label)
                
                if tokens and lemmas:
                    st.success("Lemmatization complete")
                    
                    # Display results
                    st.subheader("Results")
                    
                    # Create a table
                    result_data = {
                        "Token": tokens,
                        "Lemma": lemmas
                    }
                    st.table(result_data)
                    
                    # Download option
                    output_json = json.dumps(
                        {"tokens": tokens, "lemmas": lemmas},
                        ensure_ascii=False,
                        indent=2
                    )
                    st.download_button(
                        label="Download Results (JSON)",
                        data=output_json,
                        file_name="lemmatization_results.json",
                        mime="application/json"
                    )
                else:
                    st.warning("No valid tokens to process.")
            else:
                st.warning("Please enter some text first.")
    
    # Tab 2: Model Performance
    with tab2:
        st.header("Model Performance Comparison")
        
        if results and "comparison" in results:
            comparison = results["comparison"]
            
            # Create metrics display
            st.subheader("Performance Metrics")
            
            metrics_to_show = ["token_accuracy", "exact_match", "f1", "precision", "recall"]
            
            # Create comparison table
            table_data = []
            for model_name, metrics in comparison.items():
                row = {"Model": model_name.split('/')[-1] if '/' in model_name else model_name}
                for metric in metrics_to_show:
                    if metric in metrics:
                        row[metric] = f"{metrics[metric]:.4f}"
                table_data.append(row)
            
            st.table(table_data)
            
            # Best model highlight
            if "best_model" in results:
                st.info(f"Best performing model: {results['best_model']}")
            
            # Visualize metrics
            st.subheader("Metric Comparison")
            
            import pandas as pd
            
            chart_data = []
            for model_name, metrics in comparison.items():
                model_short = model_name.split('/')[-1] if '/' in model_name else model_name
                for metric in metrics_to_show:
                    if metric in metrics:
                        chart_data.append({
                            "Model": model_short,
                            "Metric": metric.replace('_', ' ').title(),
                            "Value": metrics[metric]
                        })
            
            if chart_data:
                df = pd.DataFrame(chart_data)
                
                for metric in metrics_to_show:
                    metric_label = metric.replace('_', ' ').title()
                    metric_df = df[df["Metric"] == metric_label]
                    if not metric_df.empty:
                        st.markdown(f"**{metric_label}**")
                        st.bar_chart(metric_df.set_index("Model")["Value"])
        else:
            st.info("No performance results available.")
    
    # Tab 3: Error Analysis
    with tab3:
        st.header("Error Analysis")
        
        if results and "errors" in results:
            errors = results["errors"]
            
            st.write(f"Total errors analyzed: {len(errors)}")
            
            # Display errors in a table
            if errors:
                # Convert Buckwalter to Arabic for display
                error_df = {
                    "Token": [buckwalter_to_arabic(e["token"]) for e in errors[:50]],
                    "Predicted": [buckwalter_to_arabic(e["pred"]) for e in errors[:50]],
                    "True Label": [buckwalter_to_arabic(e["true"]) for e in errors[:50]]
                }
                
                st.dataframe(error_df, use_container_width=True, height=400)
                
                # Error statistics
                st.subheader("Error Statistics")
                col1, col2 = st.columns(2)
                
                with col1:
                    unique_tokens = len(set([e["token"] for e in errors]))
                    st.metric("Unique Error Tokens", unique_tokens)
                
                with col2:
                    unique_lemmas = len(set([e["true"] for e in errors]))
                    st.metric("Unique True Lemmas", unique_lemmas)
                
                # Download errors
                errors_json = json.dumps(errors, ensure_ascii=False, indent=2)
                st.download_button(
                    label="Download Error Analysis (JSON)",
                    data=errors_json,
                    file_name="error_analysis.json",
                    mime="application/json"
                )
            else:
                st.info("No errors recorded in the results file.")
        else:
            st.info("No error analysis available.")
    
    # Tab 4: About
    with tab4:
        st.header("About This Project")
        
        st.markdown("""
        ### Project Overview
        This is an Arabic Lemmatization system developed as part of a Machine Learning class project.
        
        Lemmatization is the process of reducing words to their base or dictionary form (lemma).
        For example, different inflected forms of a verb are mapped to their root form.
        
        ### Objectives
        - Implement token classification for Arabic text
        - Compare multiple transformer-based models
        - Deploy a functional web application
        
        ### Models Used
        - **AraBERT v02**: Arabic BERT model pre-trained on Arabic text
        - **CAMeL-BERT MSA**: BERT model trained on Modern Standard Arabic
        - **Baseline**: Simple lookup-based model for comparison
        
        ### Dataset
        - **Source**: Quranic Corpus Morphology (v0.4)
        - **Size**: Approximately 6,200 verses with morphological annotations
        - **Split**: 80% training, 10% validation, 10% testing
        
        ### Technologies
        - Transformers (Hugging Face)
        - PyTorch
        - Streamlit
        - seqeval
        
        ### Team Information
        [Add your team information here]
        
        ### References
        - Quranic Arabic Corpus: http://corpus.quran.com/
        - AraBERT: https://github.com/aub-mind/arabert
        - CAMeL Tools: https://github.com/CAMeL-Lab/camel_tools
        """)
        
        st.markdown("---")
        st.caption("Machine Learning Class Project | 2025")

if __name__ == "__main__":
    main()