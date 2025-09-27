import streamlit as st
from transformers import pipeline

# Page setup
st.set_page_config(page_title="NLP Assistant", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #1e1e1e; color: white; }
    .block-container { padding-top: 2rem; }
    .stTextArea textarea { background-color: #1e1e1e; color: white; }
    .stSelectbox div { background-color: #1e1e1e; color: white; }
    .stButton button { background-color: #00aaff; color: white; }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🤖 NLP Assistant")
st.sidebar.write("Select a task and enter your text below.")

# Task selection
task = st.sidebar.selectbox("Choose a task", [
    "Sentiment Analysis",
    "Text Generation",
    "Translation (FR → EN)",
    "Summarization",
    "Named Entity Recognition"
])

# Text input
user_input = st.text_area("Enter your text here:", height=200)

# Run button
if st.button("Run"):
    with st.spinner("Processing..."):
        if task == "Sentiment Analysis":
            model = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
            result = model(user_input)[0]
            st.markdown(f"""
                <div style="background-color:#2a2a2a;padding:20px;border-radius:10px;">
                <h3 style="color:#00aaff;">Sentiment Analysis</h3>
                <p>{result['label']} — Score: {result['score']:.2f}</p>
                </div>
            """, unsafe_allow_html=True)

        elif task == "Text Generation":
            model = pipeline("text-generation", model="distilgpt2")
            result = model(user_input, max_length=50, do_sample=True)[0]
            st.markdown(f"""
                <div style="background-color:#2a2a2a;padding:20px;border-radius:10px;">
                <h3 style="color:#00aaff;">Generated Text</h3>
                <p>{result['generated_text']}</p>
                </div>
            """, unsafe_allow_html=True)

        elif task == "Translation (FR → EN)":
            model = pipeline("text2text-generation", model="t5-small")
            result = model(f"translate French to English: {user_input}")[0]
            st.markdown(f"""
                <div style="background-color:#2a2a2a;padding:20px;border-radius:10px;">
                <h3 style="color:#00aaff;">Translation</h3>
                <p>{result['generated_text']}</p>
                </div>
            """, unsafe_allow_html=True)

        elif task == "Summarization":
            model = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
            result = model(user_input)[0]
            st.markdown(f"""
                <div style="background-color:#2a2a2a;padding:20px;border-radius:10px;">
                <h3 style="color:#00aaff;">Summarization</h3>
                <p>{result['summary_text']}</p>
                </div>
            """, unsafe_allow_html=True)

        elif task == "Named Entity Recognition":
            model = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
            entities = model(user_input)
            st.markdown(f"""
                <div style="background-color:#2a2a2a;padding:20px;border-radius:10px;">
                <h3 style="color:#00aaff;">Named Entity Recognition</h3>
                {"".join([f"<p>{e['word']} → {e['entity_group']} (Score: {e['score']:.2f})</p>" for e in entities])}
                </div>
            """, unsafe_allow_html=True)









