import random
import json
import pickle
import numpy as np
import nltk
import tensorflow as tf
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

intents = json.loads(open(r'C:\Users\haric\OneDrive\Desktop\chatbot\intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

model = load_model('chatbot_financemodel.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.5
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])

def extract_number(message):
    # Extract the first occurrence of a number from the message
    numbers = [int(s) for s in message.split() if s.isdigit()]
    return numbers[0] if numbers else None

def calculate_total_income(income_sources):
    total_income = sum(income_sources.values())
    return total_income

def calculate_tax_old_regime(taxable_income, age):
    if age < 60:
        if taxable_income <= 250000:
            tax = 0
        elif taxable_income <= 500000:
            tax = (taxable_income - 250000) * 0.05
        elif taxable_income <= 1000000:
            tax = 12500 + (taxable_income - 500000) * 0.20
        else:
            tax = 112500 + (taxable_income - 1000000) * 0.30
    elif 60 <= age <= 80:
        if taxable_income <= 300000:
            tax = 0
        elif taxable_income <= 500000:
            tax = (taxable_income - 300000) * 0.05
        elif taxable_income <= 1000000:
            tax = 25000 + (taxable_income - 500000) * 0.20
        else:
            tax = 125000 + (taxable_income - 1000000) * 0.30
    else:
        if taxable_income <= 500000:
            tax = 0
        elif taxable_income <= 1000000:
            tax = (taxable_income - 500000) * 0.20
        else:
            tax = 100000 + (taxable_income - 1000000) * 0.30
    return tax

def calculate_tax_new_regime(taxable_income):
    if taxable_income <= 300000:
        tax = 0
    elif taxable_income <= 600000:
        tax = (taxable_income - 300000) * 0.05
    elif taxable_income <= 900000:
        tax = 15000 + (taxable_income - 600000) * 0.10
    elif taxable_income <= 1200000:
        tax = 45000 + (taxable_income - 900000) * 0.15
    elif taxable_income <= 1500000:
        tax = 90000 + (taxable_income - 1200000) * 0.20
    else:
        tax = 150000 + (taxable_income - 1500000) * 0.30
    return tax

def suggest_tax(tax_old_regime, tax_new_regime):
    if tax_old_regime < tax_new_regime:
        return "Based on Calculations ,Old tax regime is more benificial."
    elif tax_old_regime > tax_new_regime:
        return "Based on Calculations ,New tax regime is more benificial."
    else:
        return "Both Old and New Tax regime is same."

def chatbot_response(message, income_sources, deductions, age):
    income_categories = [
        "Basic salary income",
        "Annual income from other sources",
        "Annual income from interest",
        "Annual income from let-out house property (rental income)",
        "Dearness allowance (DA) received per annum",
        "HRA received per annum"
    ]
    deduction_categories = [
        "Basic deductions u/s 80C",
        "Contribution to NPS u/s 80CCD(1B)",
        "Medical Insurance Premium u/s 80D",
        "Donation to charity u/s 80G",
        "Interest on Educational Loan u/s 80E",
        "Interest on deposits in saving account u/s 80TTA/TTB"
    ]
    
    category_index = 0
    deduction_index = 0
    
    # Additional logic to handle user interaction and return responses
    # similar to the original script's while loop
    
    # For simplicity, let's assume the response is based on a single message
    if 'income' in message.lower():
        # Simulate income entry logic
        amount = extract_number(message)
        if amount is not None:
            income_sources[income_categories[category_index]] = amount
            category_index += 1
            if category_index < len(income_categories):
                return f"Please enter your annual income from {income_categories[category_index]} (if any):"
            else:
                total_income = calculate_total_income(income_sources)
                return f"Got all your income details. Your total income is {total_income}. Now, let's move on to deductions. Please enter your {deduction_categories[deduction_index]}:"
    
    elif 'deduction' in message.lower():
        # Simulate deduction entry logic
        amount = extract_number(message)
        if amount is not None:
            deductions[deduction_categories[deduction_index]] = amount
            deduction_index += 1
            if deduction_index < len(deduction_categories):
                return f"Please enter your {deduction_categories[deduction_index]}:"
            else:
                total_deductions = sum(deductions.values())+50000
                return f"Got all your deduction details. Your total deductions are {total_deductions}. Can you please tell me your age?"
    
    elif 'age' in message.lower():
        age = extract_number(message)
        if age is not None:
            total_income = calculate_total_income(income_sources)
            total_deductions = sum(deductions.values())+50000
            taxable_income = total_income - total_deductions
            tax_old_regime = calculate_tax_old_regime(taxable_income, age)
            tax_new_regime = calculate_tax_new_regime(taxable_income)
            suggestion = suggest_tax(tax_old_regime, tax_new_regime)
            return f"Based on your total income of {total_income}, total deductions of {total_deductions}, and age of {age}, your taxable income is {taxable_income}. Under the old regime, your tax is: {tax_old_regime}. Under the new regime, your tax is: {tax_new_regime}. {suggestion}"
    
    # Default response if none of the above conditions are met
    ints = predict_class(message)
    res = get_response(ints, intents)
    return res
