from pymongo import MongoClient
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Connect to MongoDB
client = MongoClient('localhost', 27017)
db = client.office_hours
collection = db.questions

# Load questions from MongoDB
questions = list(collection.find({}, {'_id': 0}))

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode all questions and context
corpus = [q['question'] + ' ' + q['context'] for q in questions]
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

def retrieve_relevant_question(query):
    # Encode the query
    query_embedding = model.encode(query, convert_to_tensor=True)
    
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