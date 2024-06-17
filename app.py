from flask import Flask, request, jsonify, render_template
from pymongo import MongoClient
from retriever import retrieve_relevant_question
from generator import generate_response

app = Flask(__name__, static_folder='static', static_url_path='')

# Connect to MongoDB
client = MongoClient('localhost', 27017)
db = client.office_hours
collection = db.questions

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/questions', methods=['GET'])
def get_questions():
    questions = list(collection.find({}, {'_id': 0}))
    return jsonify(questions)

@app.route('/response', methods=['POST'])
def get_response():
    data = request.json
    student_query = data.get('query')
    
    retrieved_question = retrieve_relevant_question(student_query)
    response = generate_response(retrieved_question, student_query)
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
