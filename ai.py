import argparse
import pandas as pd
from PIL import Image
from pdf2image import convert_from_path
import pytesseract
import os
import glob
from datetime import datetime
from gpt4all import GPT4All
import json
import re

# =============================================================================
# POPPLER PATH FIX
# =============================================================================
os.environ["PATH"] += os.pathsep + r"C:\Users\poppler-25.12.0\Library\bin"

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
MODEL_FILE = "Phi-3-mini-4k-instruct.Q4_0.gguf"
MODEL_FOLDER = r"C:\Users\gigabyte\AppData\Local\nomic.ai\GPT4All"

print("Loading AI model...")
try:
    # Attempt to use GPU (CUDA mode)
    model = GPT4All(MODEL_FILE, model_path=MODEL_FOLDER, allow_download=False, device="gpu")
    print("Model successfully loaded in GPU (CUDA) mode!\n")
except Exception as e:
    print(f"GPU (CUDA) mode initialization failed ({e}). Falling back to CPU mode...")
    # Fallback to CPU mode if GPU fails
    model = GPT4All(MODEL_FILE, model_path=MODEL_FOLDER, allow_download=False, device="cpu")
    print("Model successfully loaded in CPU mode!\n")

# =============================================================================
# DEFAULT STRUCTURE FOR OUTPUT DATA
# =============================================================================
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

# =============================================================================
# FUNCTIONS
# =============================================================================
def extract_text(file_path):
    """
    Extract text from a PDF or image using pytesseract OCR.
    Handles both PDF files (pages converted to images) and supported image types.
    """
    print(f"Reading: {os.path.basename(file_path)}")
    if file_path.lower().endswith(".pdf"):
        print("   Converting PDF...")
        images = convert_from_path(file_path)
        text = ""
        for i, img in enumerate(images):
            print(f"   OCR page {i+1}...")
            text += pytesseract.image_to_string(img) + "\n"
    else:
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
    print("OCR done.\n")
    return text


def fallback_extraction(text):
    """
    Basic fallback mechanism to extract invoice data using regex if AI fails.
    """
    print("Using fallback regex extraction...\n")
    extracted_data = {
        "InvoiceNumber": re.search(r"Invoice Number[:\s]*(\w+)", text, re.IGNORECASE),
        "InvoiceDate": re.search(r"Invoice Date[:\s]*([\d/.-]+)", text, re.IGNORECASE),
        "TotalAmount": re.search(r"Total[:\s]*[$]?\s*([\d,]+(?:\.\d+)?)", text, re.IGNORECASE),
    }
    # Map extracted regex results or fallback to "/"
    fallback_data = {key: match.group(1) if match else "/" for key, match in extracted_data.items()}

    # Ensure all other fields are initialized as "/"
    return {**DEFAULT_STRUCTURE, **fallback_data}


def ai_extract(text):
    """
    Use the GPT4All model (loaded with CUDA/CPU fallback) to extract structured invoice data.
    """
    print("AI extracting data...")
    
    prompt = f"""You are an invoice data extraction tool.
Return ONLY this exact JSON format. No code, no explanation, no extra text.

{{
  "InvoiceNumber": "extract or /",
  "InvoiceDate": "extract or /",
  "DueDate": "extract or /",
  "InvoiceType": "extract or /",
  "SellerName": "extract or /",
  "SellerCountry": "extract or /",
  "SellerAddress": "extract or /",
  "SellerVAT": "extract or /",
  "BuyerName": "extract or /",
  "BuyerCountry": "extract or /",
  "BuyerAddress": "extract or /",
  "BuyerVAT": "extract or /",
  "Currency": "extract or /",
  "Subtotal": "extract or /",
  "TaxAmount": "extract or /",
  "TotalAmount": "extract or /",
  "PaymentMethod": "extract or /",
  "PaymentStatus": "extract or /",
  "PaymentDate": "extract or /",
  "BankName": "extract or /",
  "IBAN": "extract or /",
  "SWIFT": "extract or /",
  "Notes": "extract or /",
  "Discount": "extract or /",
  "ShippingFee": "extract or /",
  "OtherFees": "extract or /",
  "PaymentDueDays": "extract or /"
}}

Invoice text:
{text}

Return ONLY the JSON above with filled values."""
    
    try:
        with model.chat_session():
            response = model.generate(prompt, max_tokens=600, temp=0.0)
        print("Raw AI response received.\n")
        
        # Parse the JSON output from the AI
        start = response.find("{")
        end = response.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON found")
        json_str = response[start:end]
        data = json.loads(json_str)
        print("Extraction successful!\n")
        
        # Fill missing fields with DEFAULT_STRUCTURE:
        return {**DEFAULT_STRUCTURE, **data}

    except Exception as e:
        print(f"AI model failed. Reason: {e}")
        print("Falling back to basic regex extraction.\n")
        return fallback_extraction(text)


def generate_unique_filename(base_name="archive"):
    """
    Generate unique filenames (e.g., timestamped) to avoid overwriting.
    """
    now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    return f"{base_name}_{now}"


def save_data(df, base_name, format_type):
    """
    Save the data in Excel and/or CSV file formats.
    """
    if "excel" in format_type:
        excel_file = generate_unique_filename(base_name) + ".xlsx"
        df.to_excel(excel_file, index=False)
        print(f"Created Excel: {excel_file}")
    
    if "csv" in format_type:
        csv_file = generate_unique_filename(base_name) + ".csv"
        df.to_csv(csv_file, index=False)
        print(f"Created CSV: {csv_file}")


def process_single_file(input_path, base_name, format_type):
    """
    Process a single file and save the results.
    """
    text = extract_text(input_path)
    data = ai_extract(text)
    
    df = pd.DataFrame([data])
    save_data(df, base_name, format_type)
    print(f"Processed: {os.path.basename(input_path)}\n")


def process_folder(folder_path, base_name, format_type):
    """
    Process all files in a folder and combine results as needed.
    """
    files = glob.glob(os.path.join(folder_path, "*.*"))
    files = [f for f in files if f.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png'))]
    
    if not files:
        print("No files found.")
        return
    
    all_data = []
    for file in files:
        print(f"Processing: {os.path.basename(file)}")
        text = extract_text(file)
        data = ai_extract(text)
        all_data.append(data)
    
    df = pd.DataFrame(all_data)
    save_data(df, base_name, format_type)
    print(f"\nProcessed {len(files)} files")


# =============================================================================
# COMMAND LINE
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--format", default="excel", choices=["excel", "csv", "both"])
parser.add_argument("--name", default="archive")

args = parser.parse_args()

if os.path.isfile(args.input):
    process_single_file(args.input, args.name, args.format)
elif os.path.isdir(args.input):
    process_folder(args.input, args.name, args.format)
else:
    print("Invalid path.")