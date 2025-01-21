import streamlit as st
import easyocr
import pandas as pd
import requests
import json
import re
from io import BytesIO
from PIL import Image
import numpy as np

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def extract_text_from_image(image):
    """Extracts text from a single image using EasyOCR."""
    # Convert PIL image to NumPy array
    image_array = np.array(image)
    return reader.readtext(image_array, detail=0, paragraph=True)

def preprocess_text(text):
    """Cleans and preprocesses the extracted text."""
    return ' '.join(line.strip() for line in text if line.strip())

def send_to_openai_api(processed_data, api_url, api_key):
    """Sends data to the OpenAI API for NER classification."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a Master Business Card Classifier, you are provided with the OCR output of business card in text, your job is to classify each item type, like name, phone number, email, etc. And you output them in JSON format after cleaning any irrelevant text, you think step by step before you make any decision, and your output must be confirmed, OUTPUT IS ONLY IN JSON FORMAT WITH CARD ITEMS ONLY"},
            {"role": "user", "content": processed_data}
        ]
    }
    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        return {
            "error": True,
            "status_code": response.status_code,
            "message": response.text
        }

def parse_ner_output(ner_output):
    """Parses the NER output to extract specific business card fields."""
    
    # Field mappings with company variations
    field_mappings = {
        'name': ['name', 'full_name', 'person_name'],
        'job_title': ['job_title', 'title', 'position', 'role'],
        'phone': ['phone', 'phone_number', 'mobile', 'telephone', 'contact'],
        'email': ['email', 'mail', 'email_address'],
        'website': ['website', 'web', 'url', 'site'],
        'company': ['company', 'company_name', 'organization', 'organisation', 'business', 'employer', 'firm'],
        'address': ['address', 'location', 'business_address']
    }

    try:
        raw_content = ner_output["choices"][0]["message"]["content"]
        
        # First attempt: Direct JSON parsing
        try:
            json_data = json.loads(raw_content.strip())
            parsed_data = {}
            
            # Map fields using variations
            for standard_key, variations in field_mappings.items():
                for variant in variations:
                    if value := json_data.get(variant):
                        parsed_data[standard_key.title()] = value
                        break
                        
        except json.JSONDecodeError:
            # Fallback: Regex patterns with company variations
            patterns = {
                'name': r'(?i)"(?:name|full[_\s]name|person[_\s]name)":\s*"([^"]+)"',
                'job_title': r'(?i)"(?:job[_\s]title|title|position|role)":\s*"([^"]+)"',
                'phone': r'(?i)"(?:phone[_\s]number|phone|mobile|telephone|contact)":\s*"([^"]+)"',
                'email': r'(?i)"(?:email|mail|email[_\s]address)":\s*"([^"]+)"',
                'website': r'(?i)"(?:website|web|url|site)":\s*"([^"]+)"',
                'company': r'(?i)"(?:company|company[_\s]name|organization|organisation|business|employer|firm)":\s*"([^"]+)"',
                'address': r'(?i)"(?:address|location|business[_\s]address)":\s*"([^"]+)"'
            }
            
            parsed_data = {}
            for field, pattern in patterns.items():
                if match := re.search(pattern, raw_content):
                    parsed_data[field.title()] = match.group(1).strip()

        # Clean empty values
        parsed_data = {k: v for k, v in parsed_data.items() if v}
        return parsed_data

    except Exception as e:
        st.error(f"Failed to parse NER output: {str(e)}")
        return {"error": f"NER parsing failed: {str(e)}"}

st.title("Business Card Text Extractor and Classifier")
st.sidebar.title("Upload Options")

# Upload options
upload_option = st.sidebar.radio("Upload Mode", ("Bulk Upload", "Individual Upload"))

# Upload images
uploaded_files = st.sidebar.file_uploader(
    "Upload Business Card Images", type=["png", "jpg", "jpeg"], accept_multiple_files=(upload_option == "Bulk Upload")
)

api_url = st.sidebar.text_input("OpenAI API URL")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

if uploaded_files and api_url and api_key:
    results = []
    progress_bar = st.progress(0)
    for i, uploaded_file in enumerate(uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]):
        st.info(f"Processing file: {uploaded_file.name}")
        image = Image.open(uploaded_file)

        # Extract text
        extracted_text = extract_text_from_image(image)
        st.text(f"Extracted Text: {extracted_text}")

        # Preprocess text
        cleaned_text = preprocess_text(extracted_text)

        # Send to OpenAI API
        with st.spinner("Classifying with OpenAI API..."):
            ner_output = send_to_openai_api(cleaned_text, api_url, api_key)
            if ner_output and not ner_output.get("error"):
                parsed_data = parse_ner_output(ner_output)
                if parsed_data and not parsed_data.get("error"):
                    results.append({
                        "File Name": uploaded_file.name,
                        **parsed_data
                    })
                else:
                    error_message = parsed_data.get("error", "Unknown parsing error")
                    st.error(f"Failed to parse NER output: {error_message}")
            else:
                error_message = ner_output.get("message", "Unknown error") if ner_output else "Unknown error"
                st.error(f"Failed to process data with OpenAI API. Status Code: {ner_output.get('status_code', 'N/A')} Error: {error_message}")

        # Update progress bar
        progress_bar.progress((i + 1) / len(uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]))

    # Convert results to DataFrame
    if results:
        st.success("Processing complete!")
        results_df = pd.DataFrame(results)

        # Save results to CSV
        csv_buffer = BytesIO()
        results_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        # Display results
        st.dataframe(results_df)

        # Download link
        st.download_button(
            "Download CSV Results",
            csv_buffer,
            file_name="business_card_results.csv",
            mime="text/csv",
        )
else:
    st.warning("Please upload images and provide API credentials to start.")
