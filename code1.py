from flask import Flask, render_template
import pdfplumber
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('chat.html')

if __name__ == '__main__':
    app.run(debug=True)

# Initialize the question answering model
qa_model = pipeline("question-answering")

# Download the pre-trained model for semantic similarity
nltk.download('punkt')

# Load a pre-trained sentence transformer model
sentence_model = SentenceTransformer('paraphrase-distilroberta-base-v1')

def extract_keywords(user_question):
    # Tokenize the question into words
    tokens = word_tokenize(user_question)
    # Remove punctuation and lowercase each word
    keywords = [word.lower() for word in tokens if word.isalnum()]
    return keywords

def extract_questions_and_answers_from_pdf(pdf_path):
    # Check if cached file exists
    cache_path = pdf_path + ".cache"
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as cache_file:
            return eval(cache_file.read())

    questions_and_answers = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            lines = text.split("\n")
            question = None
            answer_lines = []
            for line in lines:
                if line.strip().endswith("?"):
                    if question and answer_lines:
                        answer = " ".join(answer_lines)
                        questions_and_answers.append((question, answer))
                    question = line.strip()
                    answer_lines = []
                elif question is not None:
                    answer_lines.append(line.strip())
            if question and answer_lines:
                answer = " ".join(answer_lines)
                questions_and_answers.append((question, answer))

    # Cache the extracted questions and answers
    with open(cache_path, "w", encoding="utf-8") as cache_file:
        cache_file.write(str(questions_and_answers))

    return questions_and_answers

def filter_questions(user_question, questions_and_answers, top_k=5):
    # Extract keywords from the user question
    user_keywords = extract_keywords(user_question)

    # Calculate TF-IDF vectors for questions
    questions = [qa[0] for qa in questions_and_answers]
    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(questions)

    # Calculate TF-IDF vector for the user question
    user_question_vector = vectorizer.transform([user_question])

    # Compute cosine similarity between user question and questions in PDF
    similarities = cosine_similarity(user_question_vector, question_vectors)[0]

    # Select top-k most similar questions
    top_indices = similarities.argsort()[-top_k:][::-1]
    filtered_questions_and_answers = [questions_and_answers[i] for i in top_indices]

    return filtered_questions_and_answers

def answer_question(user_question, questions_and_answers):
    best_answer = None
    max_matches = 0

    # Extract keywords from the user question
    user_keywords = extract_keywords(user_question)

    # Iterate through each question-answer pair
    for question, answer in questions_and_answers:
        # Extract keywords from the current question
        question_keywords = extract_keywords(question)
        # Count the number of matching keywords
        matching_keywords = sum(keyword in question_keywords for keyword in user_keywords)
        # If the current answer has more matching keywords, update the best answer
        if matching_keywords > max_matches:
            max_matches = matching_keywords
            best_answer = answer

    # Return the best answer found
    if best_answer:
        return best_answer
    else:
        return "Sorry, I couldn't find an answer to that question."

def get_answer(pdf_path, user_question):  # Renamed from 'main'
    # Check if the specified PDF file exists
    if not os.path.exists(pdf_path):
        return "Error: The specified PDF file does not exist."

    # Extract questions and answers from the PDF
    questions_and_answers = extract_questions_and_answers_from_pdf(pdf_path)

    # Filter questions based on similarity
    filtered_questions_and_answers = filter_questions(user_question, questions_and_answers)
    
    # Answer the filtered questions
    return answer_question(user_question, filtered_questions_and_answers)

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')

pdf_path=r"C:\Users\AnuPuneetKomal\Desktop\Enrolify\full1.pdf"
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_answer(pdf_path,input)

if __name__ == '__main__':
    app.run()