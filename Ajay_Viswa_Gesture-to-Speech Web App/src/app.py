import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import model_from_json
import operator
import pyttsx3
import time
from PIL import Image


# Load the model architecture from the JSON file
def load_model():
    try:
        json_file = open("src/gesture-speech-model.json", "r")
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json)
        model.load_weights("src/gesture-speech-model.weights.h5")
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return None


def main():
    loaded_model = load_model()
    html_temp = """
    <div style="background-color:#FFA500; padding:10px; margin-bottom:10px;">
    <h1 style="color:white; text-align:center;">Hand Gesture Recognition Web App</h1>
    </div>
"""
    st.markdown(html_temp, unsafe_allow_html=True)


    st.sidebar.title("Pages")
    # Add a selectbox to the sidebar:
    pages=['About Web App','Gesture Control Page']
    add_pages = st.sidebar.selectbox('', pages)


    if add_pages == 'About Web App':
        html_temp2 = """
            <body style="background-color:white; padding:10px;">
            <h3 style="color:#FFA500; text-align:center;">About Gesture-to-Speech Web App</h3>
            The main aim of this application is to convert hand gestures into speech commands, providing a natural and intuitive way to interact with technology. 
            Users can perform various gestures, which are recognized by the application to trigger corresponding spoken responses, enhancing accessibility and user experience.
            
            The application utilizes your device's camera to capture hand gestures, converting them into verbal commands without the need for touch or physical interaction. 
            This innovation promotes efficiency and convenience, allowing users to control devices from a distance.
            </body>
            <div style="background-color:black; padding:10px; margin-bottom:10px;">
            <h4 style="color:white;">Gestures Implemented:</h4>
            <ul style="color:white;">
                <li>FINE - "All good"</li>
                <li>WATER - "Needs water"</li>
                <li>ENOUGH - "Stop"</li>
                <li>LIGHT-OFF - "Turn light off"</li>
                <li>LIGHT-ON - "Turn light on"</li>
                <li>FAN-OFF - "Turn fan off"</li>
                <li>FAN-ON - "Turn fan on"</li>
                <li>RESTROOM - "Needs restroom"</li>
                <li>STOP - "Stop action"</li>
                <li>THANK-YOU - "Thank you"</li>
                <li>HELP - "Needs help"</li>
                <li>NO-GESTURE - "No action"</li>
            </ul>
            <h4 style="color:white;">Prepared using:</h4>
                <ul style="color:white;">
                    <li>OpenCV</li>
                    <li>Keras</li>
                    <li>Streamlit</li>
                    <li>Tensorflow</li>
                    <li>pyttsx3 (Text-to-Speech)</li>
                </ul>
                </div>
                """ 
        st.markdown(html_temp2, unsafe_allow_html=True)

        st.sidebar.title("Made By:")
        html_temp6 = """
            <ul style="font-weight:bold;">
                <li><strong>Ajay Viswa</strong></li>
                <li>Role: <em>Developer and Data Scientist</em></li>
                <li>Email: <a href="mailto:ajayviswa22@gmail.com">ajayviswa22@gmail.com</a></li>
                <li>Skills: Python, Machine Learning, Deep Learning, Computer Vision, Gesture Recognition</li>
                <li>LinkedIn: <a href="https://www.linkedin.com/in/ajay-viswa22" target="_blank">LinkedIn Profile</a></li>
                <li>GitHub: <a href="https://github.com/ajayviswa22" target="_blank">GitHub Profile</a></li>
            </ul>
        """
        st.sidebar.markdown(html_temp6, unsafe_allow_html=True)

    elif add_pages =='Gesture Control Page':

        # Load the saved image
        image_path = "src/Gestures.png"
        image = Image.open(image_path)
        st.sidebar.title("Instructions")
        instructions = """
        1. **Start the Web Camera**: Click the "Start Web Camera" button on the main page to begin capturing video.
        2. **Perform Gestures**: Use your hands to perform gestures in front of the camera. The application recognizes specific gestures for control.
        3. **View Predictions**: The recognized gesture will be displayed on the screen, along with the corresponding action.
        4. **Stop the Camera**: Use the stop button to release the camera and exit the gesture recognition mode.
        5. **Control Actions**: Based on the recognized gesture the application will provide audio feedback 

        **Note**: Make sure your camera is working properly and that you are in a well-lit environment for the best recognition results.
        """
        st.sidebar.markdown(instructions)



        # Display the image in the sidebar
        st.sidebar.image(image, caption='Gesture Categories', use_column_width=True)


        # Initialize the text-to-speech engine
        engine = pyttsx3.init()
        # Variable to track last spoken action and time
        last_action = None
        last_speech_time = time.time()
        action_detected_time = None  # To track when the action was first detected


        left, middle, right = st.columns(3)
        with left:
            run = st.button('Start Web Camera')

        with right:
            stop =st.button('Stop Web Camera')


        FRAME_WINDOW1 = st.image([])
        FRAME_WINDOW2 = st.image([])

        if run and not stop:
            camera = cv2.VideoCapture(0)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)

            while True:
                _, frame = camera.read()
                # Simulating mirror image
                frame = cv2.flip(frame, 1)
                # Got this from collect-data.py
                # Coordinates of the ROI
                x1 = int(0.5*frame.shape[1])
                y1 = 10
                x2 = frame.shape[1]-10
                y2 = int(0.5*frame.shape[1])
                
        
                # Drawing the ROI
                # The increment/decrement by 1 is to compensate for the bounding box
                cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0),3)
                # Extracting the ROI
                roi = frame[y1:y2, x1:x2]

                # Resizing the ROI so it can be fed to the model for prediction
                roi = cv2.resize(roi, (120, 120))
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, test_image = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY)
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                FRAME_WINDOW1.image(test_image)

                # Prediction code
                result = loaded_model.predict(test_image.reshape(1, 120, 120, 1))
                prediction = {
                'FINE': result[0][0],
                'WATER': result[0][1],
                'ENOUGH': result[0][2],
                'LIGHT-OFF': result[0][3],
                'LIGHT-ON': result[0][4],
                'FAN-OFF': result[0][5],
                'FAN-ON': result[0][6],
                'RESTROOM': result[0][7],
                'STOP': result[0][8],
                'THANK-YOU': result[0][9],
                'HELP': result[0][10],
                'NO-GESTURE': result[0][11]
                }

                prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)


                    # Action and speech output based on gesture
                if prediction[0][0] == 'FINE':
                    final_label = 'FINE'
                    action = "All good"
                elif prediction[0][0] == 'WATER':
                    final_label = 'WATER'
                    action = "Needs water"
                elif prediction[0][0] == 'ENOUGH':
                    final_label = 'ENOUGH'
                    action = "Enough, stop"
                elif prediction[0][0] == 'LIGHT-OFF':
                    final_label = 'LIGHT-OFF'
                    action = "Turn light off"
                elif prediction[0][0] == 'LIGHT-ON':
                    final_label = 'LIGHT-ON'
                    action = "Turn light on"
                elif prediction[0][0] == 'FAN-OFF':
                    final_label = 'FAN-OFF'
                    action = "Turn fan off"
                elif prediction[0][0] == 'FAN-ON':
                    final_label = 'FAN-ON'
                    action = "Turn fan on"
                elif prediction[0][0] == 'RESTROOM':
                    final_label = 'RESTROOM'
                    action = "Needs restroom"
                elif prediction[0][0] == 'STOP':
                    final_label = 'STOP'
                    action = "Stop action"
                elif prediction[0][0] == 'THANK-YOU':
                    final_label = 'THANK-YOU'
                    action = "Thank you"
                elif prediction[0][0] == 'HELP':
                    final_label = 'HELP'
                    action = "Needs help"
                elif prediction[0][0] == 'NO-GESTURE':
                    final_label = 'NO-GESTURE'
                    action = "No action"

                text1 = "Gesture: {}".format(final_label)
                text2 = "Action: {}".format(action)

                cv2.putText(frame, text1 , (10, 120), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0), 1)
                cv2.putText(frame, text2 , (10, 220), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0), 1)
                FRAME_WINDOW2.image(frame)

                # Update action detection time if the action is not 'NO-GESTURE'
                if prediction[0][0] != 'NO-GESTURE':
                    if action_detected_time is None:
                        action_detected_time = time.time()  # Set detection time

                    # Convert action to speech if new or enough time has passed (2 seconds delay)
                    if action != last_action or (time.time() - last_speech_time) > 2:
                        # Check if the action has been detected for more than 1 second
                        if (time.time() - action_detected_time) > 2:
                            engine.say(action)
                            engine.runAndWait()
                            last_action = action
                            last_speech_time = time.time()
                            action_detected_time = None  # Reset the detection time after speaking
                    else:
                        # Reset action detection time if no gesture is detected
                        action_detected_time = None

                if stop:
                    break

            camera.release()
            cv2.destroyAllWindows()



if __name__ == '__main__':
    main()