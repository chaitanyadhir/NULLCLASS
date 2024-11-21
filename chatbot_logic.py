from PIL import Image
from dotenv import load_dotenv
import streamlit as st
import os
import base64
import google.generativeai as genai
from io import BytesIO

load_dotenv()  

os.environ["GOOGLE_API_KEY"] = "your api key"

# os.environ["GOOGLE_API_KEY"] = os.getenv("AIzaSyC1dfpbwe3M4OR7INPSNX_4F4zvVdGlBSM")
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Function to load the Gemini model and get responses
model = genai.GenerativeModel("gemini-1.5-flash")
chat = model.start_chat(history=[])

def get_gemini_response(question):
    response = chat.send_message(question, stream=True)
    return response

def get_image_response(input,image):
    # model=genai.GenerativeModel("gemini-1.5-flash")
    if input!="":
        response=model.generate_content([input,image])
    else:
        response=model.generate_content(image)
    return response.text



def generate_image_with_gemini(prompt):
    # Ensure you are using a model that supports image generation
    image_model = genai.GenerativeModel("imagen-3")  # Replace with the correct model if necessary
    response = image_model.generate_content(prompt)

    if response.startswith('http'):
        return response
    else:
        try:
            base64_str = response.split(",")[1] if "," in response else response
            image_data = base64.b64decode(base64_str)
            return BytesIO(image_data)
        except Exception as e:
            print("Error decoding base64 image:", e)
            return None



st.set_page_config(page_title="GEMINI CHATBOT DEMO")
st.header("Gemini Application")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)


input_text = st.text_input("Input: ", key="input")
submit = st.button("Ask the question")
submit_image = st.button("Analyze Image with Text")
submit_generate_image=st.button("generate image")


if submit_image and uploaded_file:
    response = get_image_response(input_text, image)
    st.subheader("The Response for the Image is: ")
    st.write(response)

if submit:
    response = get_gemini_response(input_text)
    st.subheader("The Response is: ")
    for chunk in response:
        st.write(chunk.text)
    st.write(chat.history)

if submit_generate_image:
    generated_image = generate_image_with_gemini(input_text)

    if isinstance(generated_image, BytesIO):
        st.subheader("Generated Image:")
        st.image(generated_image)
    elif generated_image and generated_image.startswith('http'):
        st.subheader("Generated Image:")
        st.image(generated_image)
    else:
        st.write("Error: Unable to generate image.")
