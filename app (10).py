import streamlit as st
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Set page configuration
st.set_page_config(page_title="Agro-Product Chatbot", layout="wide")

# Initialize NLTK downloads
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'question_embeddings' not in st.session_state:
    st.session_state.question_embeddings = None
if 'data' not in st.session_state:
    st.session_state.data = None

# Text preprocessing function
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered)

# Load data and model
@st.cache_data
def load_data():
    try:
        # Assuming the CSV is uploaded or available in the Streamlit environment
        data = pd.read_csv('agro_chatbot_dataset.csv')
        data['clean_question'] = data['Question'].apply(preprocess)
        return data
    except FileNotFoundError:
        st.error("Please upload the 'agro_chatbot_dataset.csv' file.")
        return None

@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Semantic search function
def get_answer(user_input, data, model, question_embeddings):
    cleaned_input = preprocess(user_input)
    user_embedding = model.encode([cleaned_input])
    similarities = cosine_similarity(user_embedding, question_embeddings)
    best_idx = np.argmax(similarities)
    best_score = similarities[0][best_idx]
    
    if best_score > 0.6:  # Confidence threshold
        response = data.iloc[best_idx]['Answer']
        category = data.iloc[best_idx]['Category']
        language = data.iloc[best_idx]['Language']
        return f"**Category:** {category} ({language})\n\n**Answer:** {response}", best_score
    else:
        return "❌ Sorry, I couldn't find a good match. Please try rephrasing.", best_score

# Load data and model once
if not st.session_state.data_loaded:
    with st.spinner("Loading data and model..."):
        st.session_state.data = load_data()
        if st.session_state.data is not None:
            st.session_state.model = load_model()
            st.session_state.question_embeddings = st.session_state.model.encode(
                st.session_state.data['clean_question'].tolist()
            )
            st.session_state.data_loaded = True

# Streamlit UI
st.title("🌾 Agro-Product Chatbot")
st.markdown("Ask questions about Seeds, Fertilizers, Animal Feeds, Pesticides, or Agro Tools in English or Swahili.")

# File uploader for dataset (optional for deployment flexibility)
uploaded_file = st.file_uploader("Upload agro_chatbot_dataset.csv if not already available", type="csv")
if uploaded_file is not None:
    st.session_state.data = pd.read_csv(uploaded_file)
    st.session_state.data['clean_question'] = st.session_state.data['Question'].apply(preprocess)
    st.session_state.question_embeddings = st.session_state.model.encode(
        st.session_state.data['clean_question'].tolist()
    )
    st.success("Dataset uploaded successfully!")

# Chat interface
if st.session_state.data is not None:
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # User input
    user_input = st.chat_input("Type your question here...")
    if user_input:
        # Display user message
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)
        
        # Get and display bot response
        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response, score = get_answer(
                        user_input,
                        st.session_state.data,
                        st.session_state.model,
                        st.session_state.question_embeddings
                    )
                    st.markdown(response)
                    # st.write(f"(Confidence score: {score:.2f})")  # Optional for debugging
        
        # Update chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Rerun to update the chat display
        st.rerun()

else:
    st.warning("Please ensure the dataset is available or upload it to start the chatbot.")