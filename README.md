# AI-Powered-Text-Analysis-Engine
AI-Powered Text Analysis Engine
A Unified Pipeline for Data Acquisition, Preprocessing, and NLP Modeling
Overview
This project brings together two complementary stages of Natural Language Processing (NLP):
1. Domain Data Preparation – focused on mining and cleaning unstructured data
2. Sentiment Modeling & Evaluation – focused on classifying and analyzing the sentiments of textual data

Together, they form an AI-Powered Text Analysis Engine capable of handling raw data inputs (webpages, PDFs) and transforming them into valuable insights using modern machine learning models.
Project Structure
Part 1: Domain Data Preparation (NLP1.ipynb)
Goal: Create a high-quality corpus from diverse and noisy text sources (COVID-related documents).

Key Features:
- Web scraping with Selenium and BeautifulSoup
- PDF extraction using PDFMiner
- Text normalization, cleaning, and tokenization
- Lemmatization and stopword removal using NLTK
- Unified corpus creation for further NLP tasks
Part 2: Sentiment Analysis and Topic Modeling (NLP2.ipynb)
Goal: Analyze user sentiment on the IMDB Movie Reviews dataset using classification and topic modeling.

Key Features:
- Feature extraction using CountVectorizer and TF-IDF
- Word embedding generation with Word2Vec (via Gensim)
- Supervised learning using Support Vector Machines (SVM)
- Topic extraction with Latent Dirichlet Allocation (LDA)
- Cross-validation with Stratified K-Fold
Technologies Used
Language: Python
Web/Data Extraction: Selenium, BeautifulSoup, PDFMiner
NLP & Preprocessing: NLTK, regex, Gensim, scikit-learn
ML Models: SVM, LDA
Feature Engineering: TF-IDF, CountVectorizer, Word2Vec
Data Handling: pandas, NumPy
Evaluation: Accuracy, Stratified K-Fold Cross-Validation
Output
- A clean, structured text corpus sourced from online COVID content
- A high-performing sentiment classifier on movie reviews
- Topic insights from real-world text using unsupervised modeling
- An integrated NLP pipeline ready to be scaled or deployed in real-world applications
How to Run
1. Clone the repository or download both notebooks.
2. Make sure Python and required packages (see requirements.txt in each folder) are installed.
3. Run domain_data_preparation.ipynb to create your cleaned corpus.
4. Run machine_learning_b.ipynb to train and evaluate your sentiment model.
5. Explore feature vectors, embeddings, and topic clusters.
Why This Matters
This project demonstrates:
- End-to-end handling of unstructured text data
- Strong grasp on both text engineering and text understanding
- Use of interpretable ML models in real-world NLP pipelines
- The ability to build custom datasets, not just consume pre-cleaned ones

