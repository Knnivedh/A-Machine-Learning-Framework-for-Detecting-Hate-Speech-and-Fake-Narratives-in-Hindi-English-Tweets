
---

# 📊 A Machine Learning Framework for Detecting Hate Speech and Fake Narratives in Hindi-English Tweets

**Team Submission for the Faux-Hate Shared Task at ICON 2024**

This repository contains the implementation of our machine learning framework developed for the **Faux-Hate Shared Task** at **ICON 2024**, aimed at detecting **hate speech** and **fake narratives** in **Hindi-English code-mixed social media text**. Our system leverages advanced preprocessing, TF-IDF-based feature engineering, ensemble classification, and class imbalance mitigation techniques to effectively handle the linguistic complexity and real-world challenges of multilingual content moderation.

🔗 [Task Website](https://icon2024.iitgn.ac.in/faux-hate.html) | 🏆 **3rd Place in Task A** | 📊 **13th Place in Task B**

---

## 📝 Abstract

We present a robust machine learning framework designed to detect hate speech and fake narratives in Hindi-English code-mixed tweets. This task poses unique challenges due to linguistic code-switching, informal expressions, transliteration, and data imbalance. Our approach employs a comprehensive pipeline involving:

- Custom text preprocessing for noisy, multilingual social media content  
- TF-IDF vectorization with unigram and bigram features  
- Ensemble classification using **Random Forest** models  
- Class imbalance correction via **SMOTE** and class weighting  

Our system achieves strong performance on both binary classification (Task A: Fake/Hate detection) and multi-class prediction (Task B: Target & Severity). The model secured **3rd place in Task A** and demonstrated competitive results in Task B. We provide fully reproducible code, detailed experimental analysis, and ethical considerations for deployment in real-world content moderation systems.

---

## 🌐 Introduction

The proliferation of **hate speech** and **fake narratives** on social media platforms has become a critical societal challenge, especially in multilingual regions like India where **Hindi-English code-mixed communication** is widespread. These hybrid texts—often filled with transliterated words, slang, and informal grammar—pose significant challenges for traditional NLP models trained on monolingual, formal language.

Existing approaches struggle with code-switching dynamics and contextual nuance. To address this gap, we participated in the **Faux-Hate Shared Task (ICON 2024)**, which focuses on detecting harmful content in code-mixed Indian social media data. We propose **Hate-FakeNet**, a machine learning framework tailored for this domain, combining effective feature engineering and ensemble learning to improve detection accuracy across diverse linguistic patterns.

Our contributions include:
- A modular, reproducible pipeline for code-mixed text classification.
- Application of SMOTE and class weighting to mitigate label imbalance.
- Performance evaluation on both binary and multi-class subtasks.
- Open-sourcing of code and documentation to support future research.

---

## 📚 Literature Survey

Our work builds upon and extends prior research in several key areas:

| Area | Key References | Relevance |
|------|----------------|---------|
| **Code-Mixed Text Processing** | Chakravarthi et al. (2020), Srivastava & Singh (2021) | Addresses challenges like transliteration, phonetic spelling, and informal grammar in Hindi-English texts |
| **Hate Speech Detection** | Davidson et al. (2017), Malmasi & Zampieri (2018) | Highlights limitations in multilingual settings; our work fills the code-mixed gap |
| **Fake News Detection** | Shu et al. (2020), Ruchansky et al. (2017) | Supports use of linguistic features (e.g., TF-IDF) and hybrid modeling strategies |
| **Feature Engineering** | Kumar et al. (2022), Ruder (2017) | Validates effectiveness of n-grams and Random Forests in text classification |
| **Class Imbalance** | Chawla et al. (2002) – SMOTE | Critical for handling underrepresented hate/fake classes |

---

## ⚙️ Methodology

### 1. Text Preprocessing
To handle noisy, informal, and mixed-language input, we applied the following preprocessing steps:
- Removal of URLs, mentions (`@user`), hashtags (`#tag`), and non-alphanumeric characters
- Tokenization into meaningful units
- Stopword removal (using multilingual stopword lists)
- Lowercasing and normalization

> *Note:* We preserved code-mixed structure rather than translating or transliterating, respecting the natural form of user-generated content.

---

### 2. Feature Extraction

We used **TF-IDF vectorization** with:
- **Unigrams and bigrams** to capture both individual terms and contextual word pairs
- **Maximum features:** 10,000 (optimized via validation)
- **Additional handcrafted feature:** Text length (number of characters), as longer texts often contain more elaborate narratives

This resulted in high-dimensional sparse feature vectors suitable for tree-based models.

---

### 3. Handling Class Imbalance

Given the skewed distribution of hate speech and fake content (minority classes), we employed two complementary strategies:

#### ✅ SMOTE (Synthetic Minority Over-sampling Technique)
- Applied only on training data to avoid data leakage
- Generated synthetic samples for minority classes (fake/hate = 1)
- Improved model sensitivity without overfitting

#### ✅ Class Weight Adjustment
- Used `class_weight='balanced'` in Random Forest and other classifiers
- Increased penalty for misclassifying minority instances
- Enhanced recall while maintaining acceptable precision

---

## 🧪 Model Architecture & Training

We evaluated multiple classifiers to identify the best-performing architecture:

| Model | Use Case | Observations |
|-------|--------|------------|
| **Random Forest** | Primary model (**Hate-FakeNet**) | Best overall performance; robust to noise and high dimensionality |
| Logistic Regression | Baseline | Fast but less accurate on complex interactions |
| SVM | Baseline | Struggled with scalability and code-mixed nuances |
| XGBoost / LightGBM | Task B (multi-class) | Competitive but slightly overfitted |
| Feed-Forward Neural Network | Exploratory | Lower performance due to limited data size |

Ultimately, **Random Forest** emerged as the most effective and stable choice.

---

## 📈 Evaluation Metrics

We used standard metrics appropriate for imbalanced datasets:

| Metric | Formula | Purpose |
|-------|--------|--------|
| **Accuracy** | `(TP + TN) / (TP + TN + FP + FN)` | Overall correctness |
| **Precision** | `TP / (TP + FP)` | Minimize false positives |
| **Recall (Sensitivity)** | `TP / (TP + FN)` | Maximize detection of harmful content |
| **Macro F1-Score** | `2 × (P × R) / (P + R)` (per class, then averaged) | Balanced measure across classes; primary metric |

> 🔍 **Macro F1** was prioritized over accuracy due to class imbalance.

---

## 📁 Dataset & Tasks

The **Faux-Hate dataset** (ICON 2024) consists of Hindi-English code-mixed tweets annotated for hate speech and misinformation.

### Dataset Split
| Set | Size | Labels |
|-----|------|--------|
| Training | 6,397 tweets | Fake, Hate, Target, Severity |
| Validation | 801 tweets | Same as above |
| Test | 801 tweets | Unlabeled (for final submission) |

---

### Task A: Binary Faux-Hate Detection
Predict two binary labels per tweet:
- `Fake`: 1 (fake narrative), 0 (real)
- `Hate`: 1 (hate speech), 0 (non-hate)

### Task B: Target and Severity Prediction
Multi-class classification:
- `Target`: {Individual (I), Organization (O), Religion (R)}
- `Severity`: {Low (L), Medium (M), High (H)}

---

## 📊 Results

### Task A: Binary Classification

| Label | Accuracy | Precision | Recall | **Macro F1** |
|-------|---------|----------|--------|-------------|
| **Hate** | 0.7575 | 0.73 | 0.79 | **0.76** |
| **Fake** | 0.7875 | 0.81 | 0.76 | **0.78** |

✅ **Rank: 3rd out of all participants in Task A**  
🎯 High recall indicates effective detection of harmful content  
📉 False positives mainly from strong emotional (but non-hateful) language  
🧠 Implicit or subtle hate speech remains challenging

---

### Task B: Multi-Class Prediction

| Subtask | Macro F1 | Rank |
|--------|----------|------|
| **Target Prediction** | 0.42 | 13th |
| **Severity Prediction** | 0.39 | 13th |

📉 Lower performance attributed to:
- Smaller per-class sample sizes
- Subjective nature of severity labeling
- Ambiguity in target identification (e.g., overlapping group references)

Despite lower scores, the model shows promise in capturing explicit cases.

---

## 🛠️ Task-Specific Models

### `Hate-FakeNet` (Task A)
- **Base Model:** Random Forest
- **Features:** TF-IDF (unigrams + bigrams) + text length
- **Imbalance Handling:** SMOTE + class weights
- **Optimized for:** High recall with controlled precision

### `Hate-FakeNet-Plus` (Task B)
- **Enhanced Version:** Random Forest + XGBoost/LightGBM experiments
- **Multi-output Strategy:** One classifier per label (Target, Severity)
- **Focus:** Granular understanding of hate dynamics

---

## 🔍 Error Analysis

### Common Misclassifications:
- **False Positives (Hate):** Strongly opinionated but non-toxic content (e.g., political criticism)
- **False Negatives (Hate):** Indirect insults, sarcasm, culturally nuanced slurs
- **Fake News Errors:** Satire misclassified as fake; legitimate misinformation missed due to subtle phrasing

### Limitations:
- Reliance on surface-level lexical features limits contextual understanding
- No use of pre-trained multilingual language models (e.g., mBERT, IndicBERT)
- Generalization may be affected by regional dialects and slang

---

## 🤝 Acknowledgments

We gratefully acknowledge:
- The **organizers of the Faux-Hate Shared Task at ICON 2024** for providing the dataset and evaluation framework.
- Our mentors and colleagues for valuable feedback and guidance.
- The open-source community for tools and libraries enabling rapid experimentation.

---

## 📦 Repository Structure

```bash
.
├── data/                   # Dataset files (train, dev, test)
├── models/                 # Trained model checkpoints
├── src/
│   ├── preprocess.py       # Text cleaning and normalization
│   ├── features.py         # TF-IDF and feature engineering
│   ├── train.py            # Model training and evaluation
│   └── predict.py          # Inference on test data
├── notebooks/              # Exploratory analysis and visualization
├── results/                # Output predictions and metrics
├── README.md               # This file
└── requirements.txt        # Python dependencies
```

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/A-Machine-Learning-Framework-for-Detecting-Hate-Speech-and-Fake-Narratives-in-Hindi-English-Tweets.git
   cd A-Machine-Learning-Framework-for-Detecting-Hate-Speech-and-Fake-Narratives-in-Hindi-English-Tweets
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Preprocess and train:
   ```bash
   python src/preprocess.py
   python src/train.py
   ```

4. Generate predictions:
   ```bash
   python src/predict.py --input data/test.csv --output results/predictions.csv
   ```

---

## 🛡️ Ethical Considerations

Automated hate speech detection must be deployed responsibly:
- Risk of **over-censorship** or bias against marginalized voices
- Cultural and contextual sensitivity is crucial
- Human-in-the-loop review recommended before enforcement actions
- Transparency in model decisions and limitations

We advocate for **accountable AI** and encourage auditing and fairness testing before real-world deployment.

---

## 📚 Citation

If you use this work or code in your research, please cite:

```bibtex
@inproceedings{fauxhate2024,
  title={A Machine Learning Framework for Detecting Hate Speech and Fake Narratives in Hindi-English Tweets},
  author={Your Name and Team Members},
  booktitle={Proceedings of the 21st International Conference on Natural Language Processing (ICON 2024)},
  year={2024},
  organization={IEEE}
}
```

*(Update with actual citation details upon publication)*

---

## 📬 Contact

For questions, collaborations, or feedback, please reach out:
📧 nivedhbhat15@gmail.com
🐙 [GitHub Profile]-https://github.com/Knnivedh

---



---


