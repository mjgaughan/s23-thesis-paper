import os
import openai
import time

openai.api_key = os.getenv("S23PARAML")
#https://platform.openai.com/docs/guides/embeddings/use-cases

# list models
models = openai.Model.list()

#define the question for the model
premise = "In Linux, there is a function parameter with the title 'foobar'." 
context = "foobar is a pointer to an integer array."
question = "foobar is only read from in its parent function, true or false?"
promptstring = premise + question
# create a completion
completion = openai.Completion.create(model="ada", prompt=promptstring)
#is it a buffering thing?
time.sleep(15)

# print the completion
print("——————————————THE MODEL RESPONDS:")
for choice in completion.choices:
    print(choice.text)
#print(completion.choices.text)