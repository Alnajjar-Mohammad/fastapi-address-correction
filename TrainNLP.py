import json
import spacy
from spacy.tokens import DocBin
from spacy.training.example import Example
from sklearn.model_selection import train_test_split

# ==============================
# 1️⃣ Load & Prepare Labeled Data
# ==============================

json_file_path = "labeled_address_data.json"

with open(json_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

texts, annotations = [], []
for item in data:
    texts.append(item["text"])
    annotations.append({
        "entities": [(a["start"], a["end"], a["label"]) for a in item["annotations"]]
    })

train_texts, test_texts, train_annotations, test_annotations = train_test_split(
    texts, annotations, test_size=0.2, random_state=42
)

# ==============================
# 2️⃣ Convert Data to spaCy Format
# ==============================

def create_spacy_data(texts, annotations):
    db = DocBin()
    nlp = spacy.blank("en")
    for text, annotation in zip(texts, annotations):
        doc = nlp.make_doc(text)
        ents = [doc.char_span(start, end, label=label, alignment_mode="contract") for start, end, label in annotation["entities"]]
        doc.ents = [span for span in ents if span is not None]
        db.add(doc)
    return db

train_db = create_spacy_data(train_texts, train_annotations)
test_db = create_spacy_data(test_texts, test_annotations)
train_db.to_disk("train.spacy")
test_db.to_disk("test.spacy")

# ==============================
# 3️⃣ Train spaCy NER Model
# ==============================

nlp = spacy.blank("en")
ner = nlp.add_pipe("ner", last=True)
labels = ["GOV", "AREA", "BLOCK", "STREET", "BUILDING"]
for label in labels:
    ner.add_label(label)

nlp.begin_training()
for epoch in range(10):
    losses = {}
    for text, annotation in zip(train_texts, train_annotations):
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, {"entities": annotation["entities"]})
        nlp.update([example], losses=losses)
    print(f"Epoch {epoch+1}, Loss: {losses}")

nlp.to_disk("address_ner_model")
print("✅ Training complete! Model saved as 'address_ner_model'.")

# ==============================
# 4️⃣ Test Model & Correct Address
# ==============================

def correct_address_format(doc):
    required_labels = ["GOV", "AREA", "BLOCK", "STREET", "BUILDING"]
    extracted = {label: "" for label in required_labels}

    for ent in doc.ents:
        extracted[ent.label_] = ent.text

    if all(extracted.values()):
        return f"Correct Address: {extracted['GOV']}, {extracted['AREA']}, {extracted['BLOCK']}, {extracted['STREET']}, {extracted['BUILDING']}"
    else:
        return "❌ Incorrect address format. Please provide a full address."

nlp_test = spacy.load("address_ner_model")
example_address = "Block 5, Hawalli, Street X, Building 10"
doc = nlp_test(example_address)
print(correct_address_format(doc))
