# Easy Business Card Scanner
## Purpose
Scan business cards using EasyOCR, and OpenAI GPT-4o-mini using API to classify text with help of patterns and Regular Expressions and export in CSV

## Procedures
- Enter your OpenAI standard or Custom URL and API Key.
- Upload image individually or bulk.
- If the OCR model is loaded, it automatically processes the image, and generates OCR text.
- GPT Model receives the OCR text, and cleans and sorts, and then classifies the card items
- The output is parsed and is generated in a CSV export and viewed through the streamlit app.
