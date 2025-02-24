import ollama

# Run a prompt using Llama 3
response = ollama.chat(model='llama3.2:1b', messages=[{'role': 'user', 'content': 'hello'}])

# Print the response
print(response['message']['content'])
