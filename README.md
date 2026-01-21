# hinglish-code-mixed-sentiment-analysis

## Project Overview
This project performs sentiment and hate speech classification on Hinglish (Hindi–English code-mixed) social media text using transformer-based multilingual models. The objective is to study how different transformer architectures handle noisy, code-mixed data and to compare their performance using robust evaluation metrics.

Two transformer models with different architectures are fine-tuned and evaluated:
- MuRIL (encoder-based multilingual model optimized for Indian languages)
- mBART (encoder–decoder multilingual transformer)

---

## Dataset Description
- **Raw Dataset (`bprism.csv`)**  
  Contains original Hinglish code-mixed text collected from social media sources with sentiment / hate labels.

- **Cleaned Dataset (`bprism_c.csv`)**  
  Preprocessed version of the raw data including text normalization, noise removal, and formatting suitable for transformer tokenization.

---

## Tools and Technologies Used

### Programming Language
- Python

### Development Environment and Platforms
- **Jupyter Notebook** was used as the primary development environment for experimentation and analysis.
- **Google Colab** was used to train and evaluate the **MuRIL** model due to its moderate GPU requirements.
- **Kaggle Notebooks** were used to train and evaluate the **mBART** model, as the encoder–decoder architecture requires higher GPU memory and computational resources.

This platform-specific setup ensured efficient utilization of available resources while maintaining consistent experimental conditions.

### Libraries and Frameworks
- PyTorch – Model training and optimization  
- Hugging Face Transformers – Pretrained MuRIL and mBART models and tokenizers  
- Scikit-learn – Evaluation metrics (accuracy, balanced accuracy, precision, recall, F1-score, AUC-ROC)  
- NumPy – Numerical operations  
- Pandas – Dataset handling and preprocessing  
- Matplotlib – Performance plots and confusion matrices  

### Models
- MuRIL – Multilingual encoder-based transformer optimized for Indian and code-mixed languages  
- mBART – Multilingual encoder–decoder transformer adapted for classification  

### Training Utilities
- AdamW optimizer  
- Linear learning rate scheduler with warmup  
- Early stopping based on validation F1-score  

### Platform
- GitHub – Project hosting and version control  

---

## Experimental Setup
- Task: Binary hate speech / sentiment classification  
- Data Split: 70% Training, 10% Validation, 30% Testing  
- Model selection based on validation F1-score  
- Early stopping applied to prevent overfitting  

---

## Model 1: MuRIL

### Architecture
MuRIL is an encoder-only multilingual transformer specifically designed for Indian languages and code-mixed text.

### Training Hyperparameters
- Learning Rate: 2e-05  
- Batch Size: 32  
- Epochs: 8  
- Optimizer: AdamW  
- Scheduler: Linear with warmup  
- Maximum Sequence Length: 128  
- Early Stopping: Patience = 2 (Validation F1-score)  

### Test Set Performance
- Accuracy: 0.7166  
- Balanced Accuracy: 0.7258  
- Precision: 0.6476  
- Recall: 0.8552  
- Specificity: 0.5965  
- F1-score: 0.7371  
- AUC-ROC: 0.8240  

### Result Interpretation
MuRIL achieves high recall and F1-score, indicating strong capability in detecting hate-related content. The balanced accuracy confirms stable performance under class imbalance, while the AUC-ROC value reflects good class separability.

---

## Model 2: mBART

### Architecture
mBART is a multilingual encoder–decoder transformer adapted for sequence classification to evaluate generative representations for sentiment analysis.

### Training Hyperparameters
- Learning Rate: 2e-05  
- Batch Size: 32  
- Epochs: 8  
- Optimizer: AdamW  
- Scheduler: Linear with warmup  
- Warmup Ratio: 0.1  
- Maximum Sequence Length: 128  
- Early Stopping: Patience = 2 (Validation F1-score)  

### Test Set Performance
- Accuracy: 0.7299  
- Balanced Accuracy: 0.7307  
- Precision: 0.6967  
- Recall: 0.7411  
- Specificity: 0.7203  
- F1-score: 0.7182  
- AUC-ROC: 0.8217  

### Result Interpretation
mBART achieves slightly higher overall and balanced accuracy, indicating more consistent class-wise predictions. Improved precision and specificity suggest better handling of non-hate samples while maintaining competitive recall.

---

## Comparative Insights
- MuRIL performs better in recall and F1-score, making it suitable when minimizing false negatives is critical.
- mBART provides a more balanced precision–recall trade-off with marginally higher overall accuracy.
- Both models demonstrate strong robustness on Hinglish code-mixed data.

---

## Key Contributions
- End-to-end pipeline for Hinglish sentiment and hate speech analysis  
- Comparative evaluation of encoder-based and encoder–decoder transformer architectures  
- Use of balanced accuracy and AUC-ROC to address class imbalance  
- Comprehensive performance analysis using confusion matrices and learning curves  

---

## Future Work
- Incorporate explainability techniques such as SHAP or attention visualization  
- Extend to multi-class sentiment classification  
- Deploy as a web-based inference system  
- Explore larger multilingual transformer architectures  

---

## License
This project is intended for academic and research use.
