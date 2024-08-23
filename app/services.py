import logging
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define a function to initialize model and tokenizer with error handling for GPU memory issues
def initialize_model_and_tokenizer():
    global device, model

    # Set the initial device to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
    
    # Load model with error handling
    try:
        model = AutoModel.from_pretrained("intfloat/multilingual-e5-large").to(device)
        logger.info(f"Model loaded successfully on {device}.")
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA out of memory during model loading. Switching to CPU.")

        # Clear CUDA cache to free up memory
        torch.cuda.empty_cache()

        # Switch to CPU
        device = torch.device("cpu")
        model = AutoModel.from_pretrained("intfloat/multilingual-e5-large").to(device)
        logger.info("Model reloaded successfully on CPU.")

    return tokenizer

# Initialize tokenizer and model
tokenizer = initialize_model_and_tokenizer()

def average_pool(last_hidden_states, attention_mask):
    # Mask out padding tokens
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def get_text_embeddings(texts):
    global model, device

    # Tokenize and move tensors to the correct device
    batch_dict = tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
    outputs = model(**batch_dict)

    # Compute embeddings using average pooling and normalize
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Convert embeddings to list format and move back to CPU for JSON serialization
    return embeddings.cpu().tolist()

def get_score_embeddings(texts):
    # Get embeddings for the input texts
    embeddings = get_text_embeddings(texts)

    # Convert embeddings to tensors and move to the correct device
    query_embeddings = torch.tensor(embeddings[:2], device=device)
    passage_embeddings = torch.tensor(embeddings[2:], device=device)

    # Compute similarity scores
    scores = (query_embeddings @ passage_embeddings.T) * 100

    # Move scores back to CPU and convert to list format for JSON serialization
    return scores.cpu().tolist()
