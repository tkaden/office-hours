import unittest
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
from sentence_transformers import util

# Connect to MongoDB
client = MongoClient('localhost', 27017)
db = client.office_hours
collection = db.questions

# Load questions from MongoDB
questions = list(collection.find({}, {'_id': 0}))

# Initialize the sentence transformer model
retriever_model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode all questions and context
corpus = [q['question'] + ' ' + q['context'] for q in questions]
corpus_embeddings = retriever_model.encode(corpus, convert_to_tensor=True)

# Load GPT model and tokenizer
generator_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
generator_model = GPT2LMHeadModel.from_pretrained('gpt2')

def retrieve_relevant_question(query):
    # Encode the query
    query_embedding = retriever_model.encode(query, convert_to_tensor=True)
    
    # Check if embeddings are empty
    if query_embedding.shape[0] == 0 or corpus_embeddings.shape[0] == 0:
        raise ValueError("Embeddings are empty. Please check the input data.")

    # Compute similarity scores
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]

    # Move tensor to CPU
    cos_scores = cos_scores.cpu()

    # Find the highest scoring question
    top_result = np.argmax(cos_scores.numpy())
    
    return questions[top_result]

def generate_response(retrieved_question, student_query):
    # Construct the input prompt for the GPT model
    prompt = (f"Question: {retrieved_question['question']}\n"
              f"Context: {retrieved_question['context']}\n"
              f"Instructions: {retrieved_question['instructions']}\n\n"
              f"Student's query: {student_query}\n"
              f"Response: ")
    
    inputs = generator_tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate response
    outputs = generator_model.generate(inputs, max_length=150, num_return_sequences=1)
    response = generator_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

class TestRAGSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Ensure the retriever model and embeddings are loaded once for all tests
        cls.retriever_model = SentenceTransformer('all-MiniLM-L6-v2')
        cls.generator_model = GPT2LMHeadModel.from_pretrained('gpt2')
        cls.generator_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        cls.questions = list(collection.find({}, {'_id': 0}))
        cls.corpus = [q['question'] + ' ' + q['context'] for q in cls.questions]
        cls.corpus_embeddings = cls.retriever_model.encode(cls.corpus, convert_to_tensor=True)

    def test_retrieve_relevant_question(self):
        query = "Can you explain the process of photosynthesis?"
        retrieved_question = retrieve_relevant_question(query)
        self.assertIn("photosynthesis", retrieved_question['question'].lower())

    def test_generate_response(self):
        retrieved_question = {
            "question": "Explain the process of photosynthesis.",
            "answer": "Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy, water, and carbon dioxide into glucose and oxygen.",
            "context": "Photosynthesis occurs in the chloroplasts of plant cells. The process consists of two main stages: the light-dependent reactions and the Calvin cycle.",
            "instructions": "Ask the student to describe the two main stages of photosynthesis and the role of chloroplasts."
        }
        student_query = "Can you explain the process of photosynthesis?"
        response = generate_response(retrieved_question, student_query)
        self.assertTrue(response.startswith("Question: Explain the process of photosynthesis."))
        self.assertIn("Context:", response)
        self.assertIn("Instructions:", response)
        self.assertIn("Response:", response)

if __name__ == '__main__':
    unittest.main()