# Chatbot
This project is a PDF question answering system that extracts questions and answers from a PDF document and provides answers to user queries based on the content of the PDF. It uses various natural language processing (NLP) techniques to achieve this.

## Installation

To run the PDF question answering system, you need to have Python installed on your system. You can install the required dependencies using pip:

```bash
pip install -r requirements.txt
```
## Usage
1. Extracting Questions and Answers from PDF:
   - Use the extract_questions_and_answers_from_pdf(pdf_path) function to extract
   questions and answers from a PDF document.
   - Provide the path to the PDF document as input to the function.
   - This function returns a list of tuples, where each tuple contains a question
     and its corresponding answer extracted from the PDF.
2. Filtering Questions:
   - Use the filter_questions(user_question, questions_and_answers, top_k)
     function to filter questions based on similarity to the user question.
   - Provide the user question, extracted questions and answers from the PDF, and
     optionally, the number of top similar questions to retrieve (top_k).
   - This function returns a list of filtered questions and their corresponding
     answers based on similarity to the user question.
3. Answering User Queries:
   - Use the answer_question(user_question, questions_and_answers) function to
     answer user queries.
   - Provide the user question and the list of questions and answers (filtered or
     unfiltered) extracted from the PDF.
   - This function returns the best answer found for the user question based on
     keyword matching.
4. Web Interface:
   - Run the Flask web application by executing the app.py file.
   - Access the web interface through your browser.
   - Enter your query in the chatbox and click 'Send' to receive the answer.
## Dependencies
   - pdfplumber: A library for extracting text from PDF documents.
   - transformers: State-of-the-art natural language processing for PyTorch and
     TensorFlow.
   - sentence-transformers: Sentence embeddings using transformer models.
   - scikit-learn: Machine learning library for Python.
   - nltk: Natural Language Toolkit for Python.
   - Flask: Web framework for building web applications in Python.
## Contributing
Contributions are welcome! If you have any suggestions, enhancements, or bug fixes, please open an issue or submit a pull request.
