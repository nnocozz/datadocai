import streamlit as st
import pandas as pd
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import tempfile
import os
import json
from datetime import datetime
from gpt4all import GPT4All

# Setting default Poppler path (adjust for your operating system)
os.environ["PATH"] += os.pathsep + r"C:\Users\poppler-25.12.0\Library\bin"

# GPT4All Model Configuration
MODEL_FILE = "Phi-3-mini-4k-instruct.Q4_0.gguf"
MODEL_FOLDER = r"C:\Users\gigabyte\AppData\Local\nomic.ai\GPT4All"

# Cache the model so it loads only once
@st.cache_resource
def load_model():
    try:
        model = GPT4All(MODEL_FILE, model_path=MODEL_FOLDER, allow_download=False, device="gpu")
        st.success("Model successfully loaded with GPU acceleration!")
        return model
    except Exception as e:
        st.warning(f"GPU initialization failed. Falling back to CPU. ({e})")
        model = GPT4All(MODEL_FILE, model_path=MODEL_FOLDER, allow_download=False, device="cpu")
        return model

# Load the model only once
model = load_model()

# Default JSON output structure
DEFAULT_STRUCTURE = {
    "InvoiceNumber": "/",
    "InvoiceDate": "/",
    "DueDate": "/",
    "InvoiceType": "/",
    "SellerName": "/",
    "SellerCountry": "/",
    "SellerAddress": "/",
    "SellerVAT": "/",
    "BuyerName": "/",
    "BuyerCountry": "/",
    "BuyerAddress": "/",
    "BuyerVAT": "/",
    "Currency": "/",
    "Subtotal": "/",
    "TaxAmount": "/",
    "TotalAmount": "/",
    "PaymentMethod": "/",
    "PaymentStatus": "/",
    "PaymentDate": "/",
    "BankName": "/",
    "IBAN": "/",
    "SWIFT": "/",
    "Notes": "/",
    "Discount": "/",
    "ShippingFee": "/",
    "OtherFees": "/",
    "PaymentDueDays": "/"
}

# Helper function to extract text from uploaded PDF or image
def extract_text(uploaded_file):
    """
    Extract text from PDF or image files. Handles PDFs using pdf2image.
    """
    if uploaded_file.name.lower().endswith(".pdf"):
        # Save uploaded file as a temporary file and process with pdf2image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(uploaded_file.getbuffer())
            temp_pdf_path = temp_pdf.name
            
        # Process the temporary PDF
        try:
            images = convert_from_path(temp_pdf_path)
            text = ""
            for img in images:
                text += pytesseract.image_to_string(img)
            return text
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            return ""
        finally:
            # Remove the temporary file
            os.remove(temp_pdf_path)
    else:
        # Process image files (JPG, PNG, etc.)
        img = Image.open(uploaded_file)
        return pytesseract.image_to_string(img)

# AI-based extraction
def ai_extract(text):
    """
    Use the GPT4All model to extract structured invoice data from text.
    """
    prompt = f"""You are an invoice data extraction tool.
    Return ONLY this exact JSON format. No code, no explanation, no extra text.

    {json.dumps(DEFAULT_STRUCTURE, indent=2)}

    Invoice text:
    {text}

    Return ONLY the JSON above with filled values."""

    try:
        with model.chat_session():  # Start a model chat session
            response = model.generate(prompt, max_tokens=600, temp=0.0)
        data = json.loads(response)
        # Fill missing values from the default structure
        return {**DEFAULT_STRUCTURE, **data}
    except Exception as e:
        st.warning(f"AI extraction failed: {e}")
        return DEFAULT_STRUCTURE

# Process the uploaded files
def process_files(uploaded_files):
    """
    Process multiple uploaded files and extract structured invoice data.
    """
    results = []
    for uploaded_file in uploaded_files:
        try:
            # Extract text from the file
            text = extract_text(uploaded_file)

            # Use AI to extract structured data
            extracted_data = ai_extract(text)

            # Add the data to results
            results.append(extracted_data)
        except Exception as e:
            st.error(f"Error processing file '{uploaded_file.name}': {e}")
    # Convert results into a DataFrame
    df = pd.DataFrame(results)
    return df

# Streamlit app starts here
st.title("Invoice Data Extraction App (Batch Processing)")
st.write("Upload multiple invoice files, select the output format, and download the extracted data.")

# File uploader for multiple PDF or image files
uploaded_files = st.file_uploader(
    "Upload your invoice files (.pdf, .jpg, .jpeg, .png)",
    type=["pdf", "jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

# Dropdown to select output format
output_format = st.selectbox("Select output format", ["CSV", "Excel"])

if uploaded_files:
    # Process the files
    with st.spinner("Processing your files..."):
        processed_data = process_files(uploaded_files)
    
    # If DataFrame is not empty
    if not processed_data.empty:
        st.success("Files processed successfully!")
        st.write("Extracted Data:")
        st.dataframe(processed_data)

        # Provide download options for processed data
        if output_format == "CSV":
            # Convert DataFrame to CSV
            csv = processed_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="invoice_data.csv",
                mime="text/csv",
            )
        elif output_format == "Excel":
            # Convert DataFrame to Excel
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as temp_excel:
                processed_data.to_excel(temp_excel.name, index=False, engine='openpyxl')
                temp_excel.seek(0)
                excel_data = temp_excel.read()
                st.download_button(
                    label="Download Excel",
                    data=excel_data,
                    file_name="invoice_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
    else:
        st.error("No data to display. Please check the uploaded files.")