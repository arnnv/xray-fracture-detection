from flask import Flask, request, jsonify
import os
import numpy as np
import tensorflow as tf
import keras
import google.generativeai as genai
from flask_cors import CORS

app = Flask(__name__)
CORS(app, support_credentials=True)

@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = 'http://localhost:5173'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

# Load models
model_elbow_frac = tf.keras.models.load_model("./weights/ResNet50_Elbow_frac.h5")
model_hand_frac = tf.keras.models.load_model("./weights/ResNet50_Hand_frac.h5")
model_shoulder_frac = tf.keras.models.load_model("./weights/ResNet50_Shoulder_frac.h5")
model_parts = tf.keras.models.load_model("./weights/ResNet50_BodyParts.h5")

categories_parts = ["Elbow", "Hand", "Shoulder"]
categories_fracture = ['fractured', 'normal']

@app.route('/predict', methods=['POST'])
def predict_api():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    file_path = f"temp/{file.filename}"

    try:
        if not os.path.exists("temp"):
            os.makedirs("temp")

        # Save the uploaded file
        file.save(file_path)

        # Prediction function for body fracture
        size = 224
        chosen_model = model_parts  # First, check if it's fractured using model_parts

        # Load and preprocess image
        temp_img = keras.preprocessing.image.load_img(file_path, target_size=(size, size))
        x = keras.preprocessing.image.img_to_array(temp_img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])

        # First prediction: Is it fractured or not?
        fracture_prediction = np.argmax(chosen_model.predict(images), axis=1)
        fracture_result = categories_fracture[fracture_prediction.item()]

        if fracture_result == 'normal':
            # If the image is not fractured, return early
            os.remove(file_path)
            return jsonify({'prediction': 'normal, no fracture detected'})

        # If fractured, check all three body parts (Elbow, Hand, Shoulder)
        models = {
            "Elbow": model_elbow_frac,
            "Hand": model_hand_frac,
            "Shoulder": model_shoulder_frac
        }

        # Initialize variables to track the best prediction
        max_prob = -1
        best_part = None

        for part, model in models.items():
            # Make prediction for each body part model
            part_prediction = np.argmax(model.predict(images), axis=1)
            part_prob = model.predict(images)[0][part_prediction.item()]

            # Compare and choose the body part with the highest probability
            if part_prob > max_prob:
                max_prob = part_prob
                best_part = part

        # Cleanup
        os.remove(file_path)

        # Return the fracture status and the body part with the highest probability
        return jsonify({'prediction': fracture_result, 'body_part': best_part})

    except Exception as e:
        # Handle any error during prediction
        return jsonify({'error': f'Error processing the request: {str(e)}'}), 500


# Configure Gemini API
genai.configure(api_key="GEMINI_API_KEY")  # Use environment variables for security

# Create the model with the required configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}


model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
)

# Initialize chat session with history (you can adjust this based on your need)
chat_session = model.start_chat(
    history=[
        {
            "role": "user",
            "parts": [
                "You are Vivi, a knowledgeable and compassionate medical assistant chatbot. Your primary function is to provide guidance and information about bone fractures. You have an extensive understanding of bone anatomy, fracture types, common causes, symptoms, and treatment options. \n\n### Initial Greeting:\n1. Greet the user warmly and ask how you can assist them.\n2. Example: \"Hello! I’m Vivi, your assistant. How may I help you today?\"\n\n### Gathering Information:\n3. After the user responds, ask for more specifics about the suspected fracture, particularly which body part they believe is injured.\n   Example: \"Can you please tell me which part of your body you think might be fractured? (e.g., wrist, ankle, leg)\"\n\n### Provide Fracture Information Based on Body Part:\n4. Once the user specifies the body part, provide detailed information about that body part, common fractures that occur in it, symptoms to look out for, and the general course of action:\n   - **For fractures:**\n     - Describe possible fracture types (e.g., closed, open, greenstick, spiral).\n     - Explain the typical causes (e.g., falls, accidents).\n     - Offer information on symptoms (e.g., swelling, pain, deformity).\n     - Suggest next steps: \"It's important to see a healthcare provider for a diagnosis.\"\n     - Offer advice for immediate care (e.g., immobilizing the area, applying ice).\n   - **For no fractures:**\n     - Reassure the user that their symptoms might not be due to a fracture.\n     - Advise seeing a doctor for confirmation if symptoms persist.\n     - Suggest ways to manage pain (e.g., rest, pain relievers).\n   \n### Example Interactions:\n- **User:** \"I think I fractured my wrist.\"\n- **Vivi:** \"Thank you for sharing. Wrist fractures often happen from falls or impacts. The symptoms may include swelling, bruising, and pain around the wrist. If this is indeed a fracture, it’s important to keep the wrist immobilized and consult an orthopedic specialist for further evaluation.\"\n  \n- **User:** \"I think I broke my leg.\"\n- **Vivi:** \"Leg fractures can result from direct trauma, like a fall or a car accident. They may be accompanied by severe pain, swelling, and an inability to move the leg. If you suspect a fracture, it's important to avoid putting weight on the leg and seek immediate medical attention.\"\n  \n- **User:** \"I'm not sure, it just hurts a lot.\"\n- **Vivi:** \"Pain in the bones can sometimes be from bruising or sprains rather than fractures. If you're unsure, it's always a good idea to visit a doctor. They can provide a proper examination to rule out any fractures or other injuries.\"\n\n### Empathy:\n- Always respond with empathy and professionalism. Avoid offering direct medical advice and recommend that users see a healthcare provider for an accurate diagnosis and treatment.\n",
            ],
        },
        {
            "role": "model",
            "parts": ["Hello! I am Vivi, your assistant. How may I help you today?"],
        }, 
    ]
)

# Route for handling chat messages
@app.route('/chat', methods=['POST'])
def chat():
    # Retrieve the user's message from the request
    user_input = request.json.get('message')
    
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    
    # Send the user's input to the chatbot and get the response
    try:
        response = chat_session.send_message(user_input)
        return jsonify({'response': response.text}), 200
    except Exception as e:
        return jsonify({"error": f"Error processing the request: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
