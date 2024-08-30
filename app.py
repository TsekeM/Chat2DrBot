import os
import time
from flask import Flask, request, jsonify
import openai
import comet_llm
from flask_cors import CORS

# Initialize Comet project with API key
comet_llm.init(project="chat2dr-openai", api_key="")

# Set your OpenAI API key
openai.api_key = ""

# Define the context for the chatbot
context_doctor = [
    {
        "role": "system",
        "content": """
You are DoctorBot, an automated service to provide medical advice based on symptoms. 
You first greet the patient, then ask how they are feeling and collect their symptoms. 
You keep asking relevant follow-up questions until you are confident in the diagnosis based on the information given. 
If you are not sure about the diagnosis, continue to ask clarifying questions. 
Be empathetic and informative but concise in your interactions. 

Here are some common diseases and their symptoms:

## Influenza (Flu):
Symptoms: Fever, chills, cough, sore throat, runny nose, muscle aches, headaches, fatigue.

## Common Cold:
Symptoms: Runny nose, congestion, sneezing, sore throat, cough, mild body aches, headache, fatigue.

## Pneumonia:
Symptoms: Cough (often producing phlegm), fever, chills, shortness of breath, chest pain, fatigue.

## Malaria:
Symptoms: Fever, chills, headache, nausea, vomiting, muscle pain, fatigue.

## Tuberculosis (TB):
Symptoms: Persistent cough (lasting more than 3 weeks), chest pain, coughing up blood, fatigue, weight loss, fever, night sweats.

## HIV/AIDS:
Symptoms: Fever, chills, rash, night sweats, muscle aches, sore throat, fatigue, swollen lymph nodes.

## Diarrheal Diseases:
Symptoms: Frequent loose or watery stools, abdominal cramps, nausea, vomiting, fever.

## Typhoid Fever:
Symptoms: High fever, weakness, stomach pain, headache, diarrhea or constipation, rash.

## Hepatitis A:
Symptoms: Fatigue, sudden nausea and vomiting, abdominal pain, clay-colored stools, loss of appetite, low-grade fever, dark urine, joint pain, jaundice.

You respond in a short, very conversational friendly style.
""",
    }
]  # accumulate messages


def get_medical_advice(user_input):
    messages = context_doctor + [{"role": "user", "content": user_input}]
    response = openai.ChatCompletion.create(model="gpt-4", messages=messages)
    return response["choices"][0]["message"]["content"]


# Example usage
user_input = "I have a fever, headache, and muscle pain."
advice = get_medical_advice(user_input)
print(advice)


def get_completion_from_messages(messages, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(model=model, messages=messages)
    return response.choices[0].message["content"]


app = Flask(__name__)
CORS(app)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message")
    context_doctor.append({"role": "user", "content": user_input})

    start_time = time.time()
    response = get_completion_from_messages(context_doctor)
    duration = time.time() - start_time

    comet_llm.log_prompt(
        prompt=user_input,
        output=response,
        duration=duration,
        metadata={
            "role": context_doctor[-1]["role"],
            "content": context_doctor[-1]["content"],
            "context": context_doctor,
            "advice_list": [],
        },
    )

    context_doctor.append({"role": "assistant", "content": response})

    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=True)
