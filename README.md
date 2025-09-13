#Multimodal Fusion for Early Detection of Sleep Disorders & Depression
A machine learning project leveraging facial, vocal, and textual cues for non-invasive mental health assessment.

##Overview
This project explores the early detection of sleep disorders and depression through the fusion of multiple data modalities. Recognizing the strong correlation between these conditions, we developed a system that analyzes textual transcripts, speech features, and facial video data to build robust predictive models. Using the EDAIC dataset, our work demonstrates that a multimodal approach, particularly one combining signals from different human expressions, can significantly outperform traditional unimodal methods in diagnostic applications.

Our best-performing ensemble model achieved 85% accuracy for depression classification and 79% accuracy for sleep disorder detection, highlighting the potential for creating scalable, accessible mental health screening tools.

##🚀 Key Features
Multimodal Data Processing: Engineered pipelines to extract and synchronize features from text, audio, and video.

Sleep Disorder Classification:

Binary: Detects the presence or absence of a sleep disorder.

Multi-class: Predicts the severity level based on clinical questionnaire scores (PHQ-8 & PCL-C).

Depression Detection: A binary classification task highly correlated with sleep patterns.

Advanced Architectures: Implemented a range of models from classical ML (Random Forest, XGBoost) to advanced deep learning networks (CNN, BiLSTM, Transformers).

State-of-the-Art Fusion: Explored and compared early and late fusion techniques to intelligently combine model predictions for enhanced accuracy.

##🛠️ Tech Stack & Methodologies
Languages & Frameworks: Python, PyTorch, TensorFlow/Keras, Scikit-learn

Core Libraries: Pandas, NumPy, Matplotlib

Data Extraction Tools: openSMILE (for audio), OpenFace (for facial features), Whisper (for transcription)

Dataset: EDAIC (Emotionally Diverse Audio-Visual Interactions Corpus)

##📊 Model Performance Highlights
This table summarizes the top accuracy achieved for our primary prediction tasks across different modalities.


Depression Detection (Binary)
Face
CNN + BiLSTM + Transformer + Attention
85.0%

###Depression Detection (Binary)
Text + Audio
Ensemble (MLP + BiLSTM) with LR Scheduler
83.3%

###Sleep Disorder Detection (Binary)
Face
MLP (Multi-Layer Perceptron)
79.0%

###Sleep Disorder Detection (Binary)
Speech
Single-layer Weighted BiLSTM
68.5%

###Sleep Disorder Detection (Binary)
Text
Random Forest (with Cross-Validation)
69.5%

Full performance details for all models and tasks can be found in the project report.

##🔬 Project Pipeline
Our system follows a structured, multi-stage pipeline:

###Input Collection: 
User provides an audio/video recording of an interview or interaction.

###Feature Extraction:

Audio is transcribed to text using Whisper.

Acoustic features (MFCC, eGeMAPS) are extracted with openSMILE.

Facial landmarks, gaze, and action units are extracted from video with OpenFace.

###Data Synchronization: 
All features are aligned to a common timeline to ensure temporal consistency.

###Model Inference: 
The synchronized data is fed into our trained unimodal and multimodal models.

###Prediction Fusion: 
An ensemble model weighs the outputs from each modality (based on prediction confidence) to generate a final, robust prediction for sleep disorder and depression risk.

##💡 Key Findings
Facial Modality is Highly Informative: Facial expressions proved to be the single most powerful predictor for both sleep disorders and depression, with the hybrid CNN-BiLSTM-Transformer model achieving the highest accuracy (85%).

Multimodal Fusion Boosts Performance: While the facial model was the strongest, combining it with audio and text cues in an ensemble framework created a more robust and reliable system.

Hybrid Models Excel: Architectures that combine CNNs (for local patterns), RNNs (for short-term temporal sequences), and Transformers/Attention (for long-range dependencies) are exceptionally effective for complex behavioral data.

##👥 Contributors
Prem Kansagra

Poojal Katiyar

Ahmad Raza

Nishita Gupta

Diwakar Prajapati
