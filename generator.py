from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load GPT model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

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
