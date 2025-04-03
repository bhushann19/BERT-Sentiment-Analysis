import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set page configuration
st.set_page_config(
    page_title="BERT Sentiment Analysis",
    page_icon="😊",
    layout="centered"
)

@st.cache_resource
def load_model():
    """Load the BERT model and tokenizer (cached to avoid reloading)"""
    model_path = "bhushann19/BERT-Sentiment-Analysis"  # Path where you saved your model
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    return model, tokenizer

def predict_sentiment(text, model, tokenizer):
    """Predict sentiment using the loaded BERT model"""
    # Tokenize the input text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    # Map prediction to sentiment label
    sentiment = "Positive" if prediction == 1 else "Negative"
    
    return sentiment, confidence

# Initialize session state for text input if it doesn't exist
if 'text_input' not in st.session_state:
    st.session_state.text_input = ""

# App title and description
st.title("BERT Sentiment Analysis")
st.markdown("""
    This application analyzes the sentiment of text using a fine-tuned BERT model.
    Enter your text below to determine if it expresses a positive or negative sentiment.
""")

# Load model (using caching for efficiency)
with st.spinner("Loading BERT model... This might take a moment."):
    model, tokenizer = load_model()

# Text input area - use session state to populate
text_input = st.text_area("Enter text to analyze:", value=st.session_state.text_input, height=150)

# Add a predict button
if st.button("Analyze Sentiment"):
    if text_input:
        with st.spinner("Analyzing..."):
            sentiment, confidence = predict_sentiment(text_input, model, tokenizer)
            
            # Display results
            st.subheader("Results")
            
            # Determine color based on sentiment
            color = "green" if sentiment == "Positive" else "red"
            
            # Display sentiment with appropriate color
            st.markdown(f"<h3 style='color: {color};'>Sentiment: {sentiment}</h3>", unsafe_allow_html=True)
            
            # Display confidence as percentage
            st.markdown(f"Confidence: {confidence*100:.2f}%")
            
            # Display confidence meter
            st.progress(confidence)
            
            # Additional interpretation
            if confidence > 0.9:
                st.success(f"The model is very confident that this text has a {sentiment.lower()} sentiment.")
            elif confidence > 0.7:
                st.info(f"The model is reasonably confident that this text has a {sentiment.lower()} sentiment.")
            else:
                st.warning("The model is not very confident about this prediction. The text might have mixed sentiment.")
    else:
        st.warning("Please enter some text to analyze.")

# Add example texts for easy testing
st.subheader("Try with example texts")
examples = [
    "I absolutely loved this movie! The actors were amazing and the plot was fantastic.",
    "This product is terrible. It broke after just one day of use and customer service was unhelpful.",
    "The restaurant was okay. Food was good but the service was slow."
]

# Create columns for example buttons
cols = st.columns(len(examples))

# Add example buttons
for i, example in enumerate(examples):
    if cols[i].button(f"Example {i+1}"):
        # Update session state and the text area will be updated on the next rerun
        st.session_state.text_input = example
        # Use rerun instead of experimental_rerun
        st.rerun()

# Add model information
st.sidebar.header("About the Model")
st.sidebar.markdown("""
    This application uses a BERT model fine-tuned on sentiment analysis.
    
    **Model Details:**
    - Base: BERT-base-uncased
    - Fine-tuned for binary sentiment classification
    - Trained on [Your dataset name]
    
    **Performance Metrics:**
    - Accuracy: [Your accuracy]
    - F1 Score: [Your F1 score]
""")

# Footer with credits
st.markdown("---")
st.markdown("Created with Streamlit and Hugging Face Transformers")
