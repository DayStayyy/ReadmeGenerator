import ollama
from ollama import ChatResponse


# print(ollama.list())
# print(ollama.show('llama3.2'))
#
# # Test simple
response = ollama.chat(model='llama3.2:latest', messages=[
  {
    'role': 'user',
    'content': 'Generate a Python project description',
  },
])
print(response['message']['content'])