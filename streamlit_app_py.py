!pip install streamlit
!pip install pillow
!pip install easyocr

import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
import easyocr
import pandas as pd

# Load the trained model and tokenizer
model_path = "./layoutlm_sroie_final"
from transformers import LayoutLMForTokenClassification, LayoutLMTokenizerFast
model = LayoutLMForTokenClassification.from_pretrained(model_path)
tokenizer = LayoutLMTokenizerFast.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Initialize EasyOCR reader
reader = easyocr.Reader(['en']) # or specify your language(s)

# Label mapping
id2label = model.config.id2label
label2id = model.config.label2id

def normalize_bbox(bbox, width, height):
    """Scale coordinates to [0, 1000] range"""
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height))
    ]

def process_image(image_path):
    """Processes an image, performs OCR, and predicts NER tags."""
    img = cv2.imread(image_path)
    if img is None:
        st.error("Could not load image.")
        return None, None, None, None

    h, w, _ = img.shape

    # Perform OCR using EasyOCR
    results = reader.readtext(image_path, paragraph=False)

    words = []
    bboxes = []
    for (bbox, text, prob) in results:
        # bbox format from easyocr is [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        min_x, min_y = min(x_coords), min(y_coords)
        max_x, max_y = max(x_coords), max(y_coords)
        words.append(text)
        bboxes.append([min_x, min_y, max_x, max_y])

    if not words:
        st.warning("No text found in the image.")
        return img, [], [], []

    # Normalize bboxes
    norm_bboxes = [normalize_bbox(bbox, w, h) for bbox in bboxes]

    # Tokenize and prepare for model
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_offsets_mapping=True,
        return_token_type_ids=True,
        return_attention_mask=True
    )

    # Align bounding boxes and labels (with dummy -100 for special tokens)
    word_ids = encoding.word_ids()
    aligned_bboxes = []
    for word_idx in word_ids:
        if word_idx is None:
            aligned_bboxes.append([0, 0, 0, 0]) # Dummy box for special tokens
        else:
            aligned_bboxes.append(norm_bboxes[word_idx])

    # Add padding for bbox if needed
    padding_len = 512 - len(aligned_bboxes)
    aligned_bboxes += [[0,0,0,0]] * padding_len

    # Prepare model inputs
    input_ids = torch.tensor([encoding['input_ids']]).to(device)
    attention_mask = torch.tensor([encoding['attention_mask']]).to(device)
    token_type_ids = torch.tensor([encoding['token_type_ids']]).to(device)
    bbox_tensor = torch.tensor([aligned_bboxes], dtype=torch.long).to(device)


    # Get predictions
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            bbox=bbox_tensor,
        )

    predictions = torch.argmax(outputs.logits, dim=2)
    predicted_token_ids = predictions[0].tolist()

    # Map predicted IDs back to labels, aligning with original words
    predicted_labels = []
    token_to_word_map = {} # Map from token index to original word index
    for token_idx in range(len(word_ids)):
        word_idx = word_ids[token_idx]
        if word_idx is not None:
             token_to_word_map[token_idx] = word_idx

    current_word_idx = -1
    for token_idx, label_id in enumerate(predicted_token_ids):
        if token_idx in token_to_word_map:
             word_idx = token_to_word_map[token_idx]
             if word_idx != current_word_idx: # Only add label for the first token of a word
                 predicted_labels.append(id2label[label_id])
                 current_word_idx = word_idx

    # Ensure the number of labels matches the number of original words
    # This is a simplification; a more robust approach would align B/I/O tags correctly
    # with subtokens and then reconstruct word-level tags.
    # For now, we take the label of the first subtoken for each word.
    final_predicted_labels = []
    word_map = {}
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is not None and word_idx not in word_map:
            word_map[word_idx] = predicted_token_ids[token_idx]

    # Reconstruct labels based on original words order
    for i in range(len(words)):
        if i in word_map:
             final_predicted_labels.append(id2label[word_map[i]])
        else:
             final_predicted_labels.append('O') # Default to O if no token maps back


    return img, words, bboxes, final_predicted_labels

# --- Streamlit Interface ---
st.title("SROIE Receipt Information Extraction")

uploaded_file = st.file_uploader("Upload a receipt image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp_receipt.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, caption="Uploaded Receipt", use_column_width=True)

    if st.button("Extract Information"):
        with st.spinner("Processing..."):
            img, words, bboxes, predicted_labels = process_image("temp_receipt.jpg")

            if img is not None:
                # Draw bounding boxes with predicted labels on the image
                img_display = img.copy()
                for i, (bbox, word, label) in enumerate(zip(bboxes, words, predicted_labels)):
                    x1, y1, x2, y2 = bbox
                    # Draw rectangle (color based on label type - basic example)
                    color = (0, 255, 0) # Green for "O"
                    if label.startswith('B-') or label.startswith('I-'):
                         color = (0, 0, 255) # Red for entities

                    cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 2)
                    if label != 'O': # Draw label only for identified entities
                         cv2.putText(img_display, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, color, 1, cv2.LINE_AA)


                st.subheader("Receipt with Extracted Entities")
                st.image(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB), use_column_width=True)

                # Display extracted information in a structured format
                st.subheader("Extracted Data")
                extracted_data = []
                current_entity = None
                current_entity_text = []

                for word, label in zip(words, predicted_labels):
                    if label.startswith('B-'):
                        if current_entity_text: # Save previous entity
                             extracted_data.append({'Field': current_entity, 'Value': " ".join(current_entity_text)})
                        current_entity = label[2:] # Remove 'B-'
                        current_entity_text = [word]
                    elif label.startswith('I-') and current_entity is not None and label[2:] == current_entity:
                         current_entity_text.append(word)
                    else: # 'O' tag or start of new, unrelated token
                        if current_entity_text: # Save previous entity
                             extracted_data.append({'Field': current_entity, 'Value': " ".join(current_entity_text)})
                        current_entity = None
                        current_entity_text = []

                # Add the last entity if the list ends with one
                if current_entity_text:
                    extracted_data.append({'Field': current_entity, 'Value': " ".join(current_entity_text)})

                if extracted_data:
                    df = pd.DataFrame(extracted_data)
                    st.dataframe(df)
                else:
                    st.info("No specific entities extracted.")

        # Clean up temporary file
        os.remove("temp_receipt.jpg")

