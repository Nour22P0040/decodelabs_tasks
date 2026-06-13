import os
from google import genai

client = genai.Client(api_key="AQ.Ab8RN6JNS4Y6qAObzQTZsR6fKRBfLoVyWj9F7ogiPWm0P5rfNg")

responses = {
    "hello": "Hello! I'm DecoBot. How can I assist you today?",
    "hi": "Hey there! What can I help you with?",
    "hey": "Hey! What's on your mind?",
    "good morning": "Good morning! Ready to build something great today?",
    "good evening": "Good evening! How can I help?",
    "who are you": "I'm DecoBot, a Rule-Based AI built at DecodeLabs.",
    "what are you": "I'm a deterministic chatbot. My logic is transparent — no black box here.",
    "your name": "My name is DecoBot.",
    "what can you do": "I can answer predefined questions with 100% consistency and zero hallucination risk.",
    "help": "I can answer questions about myself, greet you, or discuss basic topics. Try asking: 'who are you'.",
    "decodelabs": "DecodeLabs is an industrial training program building the next generation of AI engineers!",
    "project 1": "Project 1 is the Rule-Based Chatbot — mastering deterministic control flow before probabilistic AI.",
    "bye": "Goodbye! Keep building!",
    "goodbye": "See you next time. Stay curious!",
    "see you": "Take care!",
    "thanks": "You're welcome! Any other questions?",
    "thank you": "Happy to help! Anything else?",
}

EXIT_COMMANDS = {"exit", "quit", "stop", "q"}


def call_llm_fallback(user_prompt):
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"You are DecoBot, an AI assistant. Answer concisely: {user_prompt}"
        )
        return response.text.strip()
    except Exception:
        return "I'm having trouble connecting right now. Try rephrasing or type 'help'."


while True:
    raw_input_string = input("\nYou: ")
    clean_input = raw_input_string.lower().strip()

    if clean_input in EXIT_COMMANDS:
        print("\nBot: Session terminated. Goodbye!\n")
        break

    if not clean_input:
        print("Bot: (silence detected — please type something)")
        continue

    reply = responses.get(clean_input)

    if reply is None:
        reply = call_llm_fallback(raw_input_string)

    print(f"Bot: {reply}")