import os
import base64
import random
import json
import re
from io import BytesIO
from PIL import Image
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from google.generativeai import GenerativeModel

# Load environment variables
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Load CSV files
def load_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading CSV file {file_path}: {e}")
        return None

# Create embeddings using batching and FAISS
def create_embeddings(df, embedding_model, index_file='hs_codes.index'):
    if os.path.exists(index_file):
        print("Loading precomputed embeddings...")
        index = faiss.read_index(index_file)
    else:
        embeddings = []
        batch_size = 32
        for i in range(0, len(df), batch_size):
            batch = df['description'].iloc[i:i + batch_size].tolist()
            batch_embeddings = embedding_model.encode(batch, convert_to_numpy=True)
            embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings)

        index = faiss.IndexIDMap(faiss.IndexFlatL2(embeddings.shape[1]))
        index.add_with_ids(embeddings, np.arange(len(df)))
        faiss.write_index(index, index_file)

    return index

# Search function using FAISS
def search_hs_code(description, k=2):
    query_vector = embedding_model.encode([description], convert_to_numpy=True)
    distances, indices = index.search(query_vector, k)
    results = []
    for idx in indices[0]:
        if idx != -1:
            row = df_map.iloc[idx]
            results.append({
                "hscode": row['hscode'],
                "description": row['description'],
                "section": row['section']
            })
    return results

# Encode Image to Base64
def encode_image(data):
    try:
        if isinstance(data, Image.Image):
            buffered = BytesIO()
            data.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        elif isinstance(data, bytes):
            img = Image.open(BytesIO(data))
            return encode_image(img)
    except Exception as e:
        print(f"Failed to encode image: {e}")
        return None

# Load a random image from the 'Invoices' folder
def load_random_image():
    invoice_dir = './data/invoices'  # Path to your 'Invoices' folder
    if not os.path.exists(invoice_dir):
        raise FileNotFoundError(f"{invoice_dir} folder not found.")
    
    image_files = [f for f in os.listdir(invoice_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    
    if not image_files:
        raise FileNotFoundError("No image files found in the 'Invoices' folder.")

    random_file = random.choice(image_files)
    image_path = os.path.join(invoice_dir, random_file)
    image = Image.open(image_path)
    
    return image, random_file

# Process invoice using AI
def process_invoice(image_data, model):
    encoded_image = encode_image(image_data)
    if not encoded_image:
        return []

    vision_prompt = """
    Analyze this invoice and extract the following information in JSON format:

    {
        "seller_name": "Seller Name",
        "seller_address": "Seller Address",
        "receiver_name": "Receiver Name",
        "receiver_address": "Receiver Address",
        "items": [
            {
                "name": "Item Name",
                "quantity": 1,
                "price": 10.00,
                "harmonized_code": "HS Code",
                "description": "HS Description"
            }
        ]
    }

    If any information is not found, use "None" as the value.
    Return ONLY the JSON object.
    """

    if model:
        vision_response = model.generate_content([vision_prompt, {"mime_type": "image/jpeg", "data": encoded_image}])
        response_text = vision_response.text.strip()

        try:
            json_string = re.search(r'```(?:json)?\s*({.*?})\s*```', response_text, re.DOTALL).group(1)
            vision_json = json.loads(json_string)

            items = vision_json.get("items", [])
            for item in items:
                hs_codes = search_hs_code(item.get('name', ''))
                if hs_codes:
                    item['harmonized_code'] = hs_codes[0]['hscode']
                    item['description'] = hs_codes[0]['description']
                else:
                    item['harmonized_code'] = "None"
                    item['description'] = "None"

            return vision_json

        except json.JSONDecodeError as e:
            st.error(f"Error decoding JSON: {e}")
            return {}
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            return {}

# Streamlit App
def main():
    st.title("Harmonized System Code Assistant")

    # Load data
    dataframes = {
        'df_harmonized_system': load_csv('./data/harmonized-system.csv'),
        'df_sections': load_csv('./data/sections.csv')
    }

    global df_map
    df_map = dataframes['df_harmonized_system']

    if df_map is None:
        st.error("Failed to load data. Please check the file path.")
        return

    # Load embedding model
    global embedding_model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Load embeddings with spinner
    with st.spinner("Loading and embedding data..."):
        global index
        index = create_embeddings(df_map, embedding_model)

    # Create Tabs
    tab1, tab2 = st.tabs(["HS Code Search", "AI Assistant"])

    # Tab 1: HS Code Search
    with tab1:
        search_term = st.text_input("Enter product description to search for HS codes:")
        if search_term:
            results = search_hs_code(search_term)
            st.subheader("Search Results:")
            st.json(results)

    # Tab 2: AI Assistant
    with tab2:
        st.header("AI Assistant")

        if st.button("Load New Image and Process"):
            with st.spinner("Loading and processing image..."):
                image_data, file_name = load_random_image()
                st.image(image_data, caption=f"Loaded Image: {file_name}", use_container_width=True)

                if genai_model:
                    result = process_invoice(image_data, genai_model)
                    st.subheader("Extracted Information:")
                    st.json(result)
                else:
                    st.error("Gemini API is not configured. Please provide a valid API key.")

# Load Gemini model if available
genai_model = None
if api_key:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    genai_model = genai.GenerativeModel('gemini-1.5-flash')
else:
    st.error("Gemini API key not found. Please check the .env file.")

# Run app
if __name__ == "__main__":
    main()