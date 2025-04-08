from transformers import AutoTokenizer, AutoModel
import torch

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer and model (example: MiniLM for faster inference)
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Move model to GPU if available
model.to(device)

# Check if the model is on GPU
for param in model.parameters():
    print("Model device:", param.device)
    break  # Only need to check one parameter

# Tokenize and encode a document
text = "What: The I-GUIDE Platform provides an open science and collaborative environment for geospatial data-intensive convergence research and education focused on sustainability and resilience challenges and enabled by advanced cyberGIS and cyberinfrastructure. Who: Geospatial and sustainability research and education communities. Why: Support convergent knowledge sharing and discovery through connecting diverse digital knowledge elements at scale. How: Democratize access to advanced cyberGIS & cyberinfrastructure and cutting-edge geospatial AI & data science capabilities. Uniqueness: Advanced cyberGIS and cyberinfrastructure, cutting-edge geospatial AI and data science capabilities, FAIR data principles, convergent approaches to sustainability challenges. To learn more about using the I-GUIDE Platform, check out Getting Started. To learn more about the NSF Institute for Geospatial Understanding through an Integrative Discovery Environment (I-GUIDE), explore our work, and find out out about upcoming events, check our website at: i-guide.io."
inputs = tokenizer(text, return_tensors="pt").to(device)  # Move inputs to GPU

# Move input tensors to GPU
for key in inputs:
    inputs[key] = inputs[key].to(device)

# Check if the inputs are on GPU
print("Inputs device:", inputs['input_ids'].device)  # Reference a specific tensor in the inputs dictionary

# Perform inference
with torch.no_grad():
    embeddings = model(**inputs).last_hidden_state.mean(dim=1)

# Check if the embeddings are on GPU
print("Embeddings device:", embeddings.device)

# Convert to a list and normalize (optional)
embedding_vector = embeddings[0].tolist()
print(embedding_vector)

# Additional confirmation for CUDA usage
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("Using CPU")

