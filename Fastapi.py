from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import spacy
import re
import os

nlp_test = spacy.load("en_core_web_md")

corrections = {
    # Street variations
    "st": "Street", "str": "Street", "stre": "Street", "stre." : "Street",
    "ste": "Street", "stt": "Street", "strt": "Street", "stree": "Street",
    "stret": "Street", "Street": "Street",

    # Block variations
    "blk": "Block", "blok": "Block", "bock": "Block", "blck": "Block",
    
    # Building variations
    "bldg": "Building", "bld": "Building", "bldgg": "Building",
    "bldng": "Building", "blding": "Building", "bldn": "Building",
    
    # Government variations
    "gov": "Government", "govt": "Government", "gvt": "Government",
    "govenment": "Government", "governmnt": "Government", "Government": "Government",

    # Area variations
    "area": "Area", "ar": "Area", "aera": "Area",

    # Road variations
    "rd": "Road", "rdd": "Road", "r0d": "Road", "roaad": "Road",
    "raod": "Road", "rood": "Road",

    # Governorate variations
    "govr": "Governorate", "govrt": "Governorate", "governor": "Governorate",
    "governrt": "Governorate", "Governorate": "Governorate",
}


# FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

class AddressInput(BaseModel):
    input_text: str

def correct_address_format(address):
    address = re.sub(r'(\D)(\d)', r'\1 \2', address)
    address = re.sub(r'(\d)(\D)', r'\1 \2', address)
    abbreviations = "|".join(corrections.keys())
    address = re.sub(rf'(\w)({abbreviations})\b', r'\1 \2', address, flags=re.IGNORECASE)

    doc = nlp_test(address)

    corrected_tokens = []
    for token in doc:
        word = token.text.strip()
        word_lower = word.lower()

        if word_lower in corrections:
            word = corrections[word_lower]

        if not any(char.isdigit() for char in word):
            word = word.capitalize()

        corrected_tokens.append(word)

    corrected_address = " ".join(corrected_tokens)

    return corrected_address

def extract_phone_and_address(input_text):
    phone_pattern = r'(\+965\s?)?(965\s?)?([9|6|5|4]\d{7})' 
    phone_match = re.search(phone_pattern, input_text)
    
    if phone_match:
        phone_number = phone_match.group(0)
        address = input_text.replace(phone_number, "").strip()
    else:
        phone_number = None
        address = input_text.strip()

    corrected_address = correct_address_format(address)

    return phone_number, corrected_address


# API endpoint to get phone number and corrected address
@app.post("/correct-address/")
def correct_address(input: AddressInput):
    phone_number, corrected_address = extract_phone_and_address(input.input_text)
    return {"phone_number": phone_number, "corrected_address": corrected_address}

# Run the app
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
