# Week2_Cellula
#Image Captioning & Text Classification App

A Streamlit application for generating image captions, classifying text, and demonstrating model quantization to reduce large model sizes like BERT or LLaMA.

Features

Image Captioning

Upload an image and generate captions using BLIP-1/BLIP-2.

Modular implementation: imagecaption.py.

Text Classification

Classify user input text with:

LSTM

LLaMA Guard

Fine-tuned DistilBERT/ALBERT with LoRA

CSV Database

Stores all inputs (text or captions) and classification results.

Auto-updates with every submission.

View the database anytime from the app.

Model Quantization

Reduce large model size (e.g., BERT, LLaMA) using int8 quantization.

Minimal accuracy loss, ~4x smaller model.

Demo included in quantization_demo.ipynb.
