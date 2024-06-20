from transformers import LlamaForCausalLM, LlamaTokenizer
from dotenv import load_dotenv
import os

load_dotenv()
llama_path = '/Users/tkaden/git/Llama3/Meta-Llama-3-8B'

# Load LLaMA model and tokenizer
tokenizer = LlamaTokenizer.from_pretrained(llama_path)
model = LlamaForCausalLM.from_pretrained(llama_path)

def generate_response(retrieved_question, student_query):
    # Construct the input prompt for the GPT model
    prompt = (f"Question: {retrieved_question['question']}\n"
              f"Context: {retrieved_question['context']}\n"
              f"Instructions: {retrieved_question['instructions']}\n\n"
              f"Student's query: {student_query}\n"
              f"Response: ")
    
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate response
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response
