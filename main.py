import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import tempfile, os, cv2, base64
import numpy as np
from util import *

set_background('backgrounds/img4.jpg')

# Function to load the model
def load_model(model_path):
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    return model

# Function to detect and crop faces from an image
def detect_and_crop_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image_np = np.array(image)
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
    faces = face_cascade.detectMultiScale(image_rgb, scaleFactor=1.1, minNeighbors=5)
    cropped_faces = [image_np[y:y+h, x:x+w] for (x, y, w, h) in faces]
    return cropped_faces

# Function to process video and detect faces
def process_video(file, num_frames=16):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(file.read())
    video_path = tfile.name

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Open the video file
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        st.error('Error: Cannot open video file')
        return []

    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_extract = num_frames
    interval = max(1, total_frames // frames_to_extract)

    cropped_images = []
    frame_count = 0
    success = True
    while success:
        frame_id = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_id % interval == 0:
            success, image = vidcap.read()
            if not success:
                break
            # Convert the image from BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces = face_cascade.detectMultiScale(image_rgb, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in faces:
                crop_face = image_rgb[y:y+h, x:x+w]
                cropped_images.append(crop_face)  # Save cropped face in list
            frame_count += 1
            if frame_count >= frames_to_extract:
                break
        else:
            success, _ = vidcap.read()

    vidcap.release()
    return cropped_images

IMG_SIZE = 224  # Define the image size as required by your model

# Function to process the image
def process_image(image, img_size=224):
    # Convert PIL image to NumPy array
    image_np = np.array(image)
    # Normalize the data from 0-255 to 0-1
    image_np = image_np / 255.0
    # Resize the image to the required dimensions
    image_resized = tf.image.resize(image_np, size=(img_size, img_size))
    # Convert the image to a TensorFlow tensor
    image_tensor = tf.convert_to_tensor(image_resized, dtype=tf.float32)
    # Add a batch dimension explicitly
    image_tensor = tf.expand_dims(image_tensor, axis=0)
    return image_tensor

# Function to classify images
def classify(image, model):
    image_tensor = process_image(image)  # Process the image to add batch dimension
    score = model.predict(image_tensor)  # Predict using the model
    return score

# Load model
model_path = "C:/Users/punee/MLProjs/DeepFake/models/24_04_2024_07_36_27-All-Images-EfficinetNetV2-Adam.h5"
model = load_model(model_path)

st.title('Deepfake Detector')

file = st.file_uploader('Upload an image or video', type=['jpeg', 'jpg', 'png', 'mp4'])

if file is not None:
    file_type = file.type.split('/')[0]
    if file_type == 'image':
        # Save the image temporarily
        image = Image.open(file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Detect and crop faces
        cropped_faces = detect_and_crop_faces(image)
        
        if cropped_faces:
            st.markdown('<div style="display: flex; overflow-x: auto;">', unsafe_allow_html=True)
            for idx, face in enumerate(cropped_faces):
                face_image = Image.fromarray(face)
                score = classify(face_image, model)
                label = 'Deepfake' if score[0][0] > 0.5 else 'Real'
                color = 'red' if score[0][0] > 0.5 else 'green'
                st.markdown(f'''
                    <div style="flex: 0 0 auto; margin-right: 10px; text-align: center;">
                        <img src="data:image/jpeg;base64,{base64.b64encode(cv2.imencode('.jpg', cv2.cvtColor(face, cv2.COLOR_RGB2BGR))[1]).decode()}" style="max-width: 200px; display: block; margin: auto;">
                        <p style="color:{color};">{label}</p>
                        <p>Chance of Deepfake: {score[0][0] * 100:.2f}%</p>
                    </div>
                ''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.write('No faces detected in the uploaded image.')

    elif file_type == 'video':
        scores = []
        cropped_images = process_video(file)
        
        st.write('Extracted and Cropped Faces:')
        st.markdown('<div style="display: flex; overflow-x: auto;">', unsafe_allow_html=True)
        
        for idx, img in enumerate(cropped_images):
            score = classify(Image.fromarray(img), model)
            scores.append(score[0][0])
            
            # Encode image to display in HTML
            _, img_encoded = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            img_base64 = base64.b64encode(img_encoded).decode('utf-8')
            
            label = 'Deepfake' if score[0][0] > 0.5 else 'Real'
            color = 'red' if score[0][0] > 0.5 else 'green'
            
            if idx % 5 == 0:
                st.markdown(f'''
                    <div style="flex: 0 0 auto; margin-right: 10px; text-align: center;">
                        <img src="data:image/jpeg;base64,{img_base64}" style="max-width: 200px; display: block; margin: auto;">
                        <p style="color:{color};">{label}</p>
                        <p>Confidence: {score[0][0] * 100:.2f}%</p>
                    </div>
                ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        avg = sum(scores) / len(scores)
        st.write('Chance of Deepfake: is {:.2f}%'.format(avg * 100))
        if avg > 0.5:
            st.write(':red[The video is a deepfake]')
        else:
            st.write(":green[The video is real]")
