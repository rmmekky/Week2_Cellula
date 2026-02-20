from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

#Fine-Tuned DistilBERT (Sentiment Model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

model.eval()

def classify_text(text, input_type="Text"):


    if input_type == "Image Caption":
        return "Neutral"

    #  Normal Text Classification
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class_id = torch.argmax(logits).item()

    label = model.config.id2label[predicted_class_id]

    # Normalize label
    label = label.capitalize()  # Positive / Negative

    return label
