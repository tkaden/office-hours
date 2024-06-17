import sys
from pymongo import MongoClient
from retriever import retrieve_relevant_question
from generator import generate_response

def load_questions():
    # Connect to MongoDB
    client = MongoClient('localhost', 27017)
    db = client.office_hours
    collection = db.questions

    # Load questions from MongoDB
    questions = list(collection.find({}, {'_id': 0}))
    return questions

def main():
    questions = load_questions()
    question_index = 0
    total_questions = len(questions)
    
    while question_index < total_questions:
        current_question = questions[question_index]
        student_query = current_question['question']

        # Display the question to the student
        print(f"Question {question_index + 1}/{total_questions}: {student_query}")
        print("Type 'next question' to move on to the next question.")
        
        # Retrieve the relevant question details from the database
        retrieved_question = retrieve_relevant_question(student_query)

        while True:
            student_input = input("Your input: ").strip().lower()

            if student_input == 'next question':
                question_index += 1
                break
            else:
                # Generate a response based on the student's input
                response = generate_response(retrieved_question, student_input)
                print(f"Response: {response}")

        if question_index >= total_questions:
            print("You have completed all the questions.")
            break

if __name__ == '__main__':
    main()
