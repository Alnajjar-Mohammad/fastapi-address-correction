from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import spacy
import re
import os

# Load the trained model
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
    address: str

def correct_address_format(address):
    # Fix spacing between letters and numbers (e.g., "st1" → "st 1")
    address = re.sub(r'(\D)(\d)', r'\1 \2', address)
    address = re.sub(r'(\d)(\D)', r'\1 \2', address)
    abbreviations = "|".join(corrections.keys())
    address = re.sub(rf'(\w)({abbreviations})\b', r'\1 \2', address, flags=re.IGNORECASE)

    # Tokenize and process the address
    doc = nlp_test(address)

    corrected_tokens = []
    for token in doc:
        word = token.text.strip()
        word_lower = word.lower()

        # Replace abbreviations with full words
        if word_lower in corrections:
            word = corrections[word_lower]

        # Capitalize first letter of place names (heuristic: words before first number)
        if not any(char.isdigit() for char in word):
            word = word.capitalize()

        corrected_tokens.append(word)

    # Join corrected tokens to form the final address
    corrected_address = " ".join(corrected_tokens)

    return corrected_address


# API endpoint to get corrected address
@app.post("/correct-address/")
def correct_address(input: AddressInput):
    corrected_address = correct_address_format(input.address)
    return {"corrected_address": corrected_address}

# Run the app
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)