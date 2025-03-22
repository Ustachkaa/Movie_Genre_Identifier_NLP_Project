# NLP Project: Simple Movie Genre Identifier

## Overview

This project leverages Natural Language Processing (NLP) techniques to classify movie genres based on their textual descriptions (synopses). By analyzing movie data, the model predicts genres, which can be useful for improving recommendation systems, content categorization, and metadata enrichment.

The project is implemented in Python and uses popular NLP libraries such as TensorFlow, Keras, and Scikit-learn. It is designed to be easy to use, with a focus on simplicity and effectiveness.

---

## Features

- **Text Preprocessing**: Cleans and prepares movie descriptions (synopses) for model input using techniques like tokenization, stopword removal, and stemming/lemmatization.
- **Genre Classification**: Utilizes advanced NLP models, including deep learning architectures, to predict movie genres.
- **Google Colab Integration**: The project is designed to run seamlessly on Google Colab, eliminating the need for local setup.
- **Modular Code**: The code is organized into reusable functions and modules for easy customization and extension.
- **Evaluation Metrics**: Includes performance evaluation using metrics like accuracy, precision, recall, and F1-score.

---

## Quick Start (Google Colab)

You can directly run the project in Google Colab without setting up a local environment:

1. **Open the Notebook**:  
   Click the "Open in Colab" button below to launch the notebook in Google Colab:  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ustachkaa/Movie_Genre_Identifier_NLP_Project/blob/main/Movie_genre_identifier_.ipynb)

2. **Run the Notebook**:  
   Follow the step-by-step instructions in the notebook to:
   - Preprocess the movie description data.
   - Train the genre classification model.
   - Evaluate the model's performance.

3. **Customize**:  
   Modify the code to experiment with different models, datasets, or hyperparameters.

---

## Dataset

The project uses a dataset containing movie titles, descriptions (synopses), and corresponding genres. The dataset is preprocessed to ensure compatibility with the model.

- **Dataset Source**: The dataset is included in the repository or can be downloaded from [Hugging Face](https://huggingface.co/) or other open data sources.
---

## Dependencies

The project requires the following Python libraries:

- TensorFlow
- Keras
- Scikit-learn
- Pandas
- Numpy
- NLTK or SpaCy (for text preprocessing)
- Matplotlib/Seaborn (for visualization)

To install the dependencies, run:

```bash
pip install tensorflow keras scikit-learn pandas numpy nltk matplotlib seaborn
```

## How to Use

1. **Clone the Repository**:  
   Clone the repository to your local machine or Google Colab environment:

   ```bash
   git clone https://github.com/Ustachkaa/Movie_Genre_Identifier_NLP_Project.git
   cd Movie_Genre_Identifier_NLP_Project
   ```
2. **Run the Notebook**:  
   Open the `Movie_Genre_Identifier_NLP_Project.ipynb` notebook in Google Colab or Jupyter Notebook and execute the cells.

3. **Preprocess Data**:  
   Use the provided functions to clean and preprocess the movie descriptions.

4. **Train the Model**:  
   Train the genre classification model using the preprocessed data.

5. **Evaluate Performance**:  
   Evaluate the model's performance using metrics like accuracy, precision, recall, and F1-score.

6. **Make Predictions**:  
   Use the trained model to predict genres for new movie descriptions.

---

## Results

The model achieves competitive performance in genre classification, with evaluation metrics displayed in the notebook. You can visualize the results using confusion matrices and classification reports.
