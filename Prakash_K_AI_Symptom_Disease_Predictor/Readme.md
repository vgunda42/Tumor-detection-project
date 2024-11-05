# AI-Driven Symptom Analysis and Disease Prediction Using BERT

## Overview

This project leverages BERT (Bidirectional Encoder Representations from Transformers) for building a predictive model that analyzes symptoms and suggests potential diseases. Designed for healthcare applications, the model takes a list of symptoms as input and outputs probable disease diagnoses, enhancing efficiency in preliminary diagnosis processes.

## Objective

To create an AI-based model that:

Accurately predicts potential diseases based on symptoms.

Assists in early diagnosis, reducing time and effort in symptom analysis.

Demonstrates BERT’s capability in understanding medical terminology through fine-tuning on a healthcare-specific dataset.
Dataset

The model is trained on a disease and symptoms dataset sourced from Kaggle, containing a comprehensive list of symptoms mapped to corresponding diseases.

## Model Architecture

The model uses BERT for sequence classification, a fine-tuned version of bert-base-uncased with an additional classification layer for predicting disease labels.

## Technical Implementation

### Data Preprocessing:

Symptom data is preprocessed and encoded, merging all symptom text data into a combined feature.
Labels are encoded as integer classes for multi-class classification.

### Model Fine-Tuning:

A pre-trained BERT model is fine-tuned on the symptoms dataset for classification.
Hyperparameters such as learning rate, batch size, and epochs are optimized for performance.

### Training and Evaluation:

Model is trained with Trainer API from Hugging Face's Transformers library, with periodic evaluation for performance monitoring.

Evaluation Strategy ensures that the best model (based on lowest evaluation loss) is saved and loaded for testing.

## Key Features

BERT-based Classifier: Utilizes a transformer model for accurate symptom understanding.

Multi-class Disease Prediction: Predicts from a broad spectrum of diseases.

Optimized for Performance: Implements evaluation strategy to load the best performing model.

User-Friendly Interface: Can be adapted into a simple web interface for real-time usage in healthcare settings.

## Results

The model demonstrates strong performance in classifying diseases based on input symptoms. It achieves notable accuracy in mapping symptom patterns to correct diseases, showing potential as a supporting tool for healthcare professionals.

## Learning Outcomes

This project highlights the effectiveness of using BERT for medical data and the potential of AI in automating disease prediction, which can aid in timely diagnosis and efficient healthcare delivery.

## Conclusion

The AI-Driven Symptom Analysis and Disease Prediction Using BERT project showcases the ability to adapt NLP models for medical applications, paving the way for AI-enhanced healthcare solutions that reduce diagnostic time and improve patient outcomes.