from cProfile import label
import json
import re
from statistics import mode
import random_responses
import pyttsx3
import nltk
import pickle
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()
import numpy
import tflearn
import tensorflow
import random
from tensorflow.python.framework import ops




# Load JSON data
def load_json(file):
    with open(file) as bot_responses:
        print("")
        print("-- Cafe BOT --")
        print("")
        return json.load(bot_responses)

# Store JSON data
response_data = load_json("bot.json")


try:
    with open("data.pickle","rb") as f:
        words,labels,training,output = pickle.load(f) #load and train only once
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in response_data:
        for pattern in intent["user_input"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["response_type"])

            if intent["response_type"] not in labels:
                labels.append(intent["response_type"])


    words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)
    
    with open("data.pickle","wb") as f:
        pickle.dump((words,labels,training,output),f)

ops.reset_default_graph()


net = tflearn.input_data(shape=[None,len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(output[0]),activation="softmax")
net = tflearn.regression(net)

try:
    model = tflearn.DNN(net)
except:
    model = tflearn.DNN(net)

try: #not to train trained model again
    # model.load("model.tflearn")
    gjygfjy

except:
    model.fit(training, output,n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")



def bag_of_words(s,words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i,w in enumerate(words):
            if w==se:
                bag[i] = 1

    return numpy.array(bag)


def chat():
    print("Start talking with the bot!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp,words)])

        results_index = numpy.argmax(results)
        tag = labels[results_index]
        
        for tg in response_data:
                
                if tg["response_type"] == tag:
                    responses = tg["bot_response"]
        print(random.choice(responses))


chat()

# def get_response(input_string):
#     split_message = re.split(r'\s+|[,;?!.-]\s*', input_string.lower())
#     score_list = []

#     # Check all the responses
#     for response in response_data:
#         response_score = 0
#         required_score = 0
#         required_words = response["required_words"]

#         # Check if there are any required words
#         if required_words:
#             for word in split_message:
#                 if word in required_words:
#                     required_score += 1

#         # Amount of required words should match the required score
#         if required_score == len(required_words):
#             # print(required_score == len(required_words))
#             # Check each word the user has typed
#             for word in split_message:
#                 # If the word is in the response, add to the score
#                 if word in response["user_input"]:
#                     response_score += 1

#         # Add score to list
#         score_list.append(response_score)
#         # Debugging: Find the best phrase
#         # print(response_score, response["user_input"])

#     # Find the best response and return it if they're not all 0
#     best_response = max(score_list)
#     response_index = score_list.index(best_response)

#     # Check if input is empty
#     if input_string == "":
#         return "Please type something so we can chat :("

#     # If there is no good response, return a random one.
#     if best_response != 0:
#         return response_data[response_index]["bot_response"]

#     return random_responses.random_string()
# engine = pyttsx3.init()

# engine.say('I am Your Online Coffe Helper,How Can I Help You')
# print("Bot: " 'I am Your Online Coffe Helper')
# print("")
# engine.runAndWait()
# while True:
#     user_input = input("You: ")
#     print("")
#     print("Bot:", get_response(user_input))
#     engine.say(get_response(user_input))
#     print("")
#     engine.runAndWait()