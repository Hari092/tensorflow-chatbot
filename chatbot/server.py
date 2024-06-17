
from flask import Flask, request, jsonify
import json
from chatbot import predict_class, get_response  

app = Flask(__name__)

with open('C:/Users/haric/OneDrive/Desktop/chatbot/intents.json') as file:
    intents = json.load(file)

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    message = data['message']
    response = get_response(predict_class(message), intents)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
