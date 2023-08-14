import cv2
import PySimpleGUI as sg
from datetime import datetime
import os
from pathlib import Path
import shutil

from faces_train import train_model
from faces_detect import detect_faces

image_layout = []
selected_files = []

def updateLayout(image_layout = []):
    image_column = [
        [sg.Text("Face Detection & Recognition", font=("Helvetica", 25), justification='center')],
        [sg.Text("By: Cayco, Fariscal, Merculio, Morales, Paclian, Villar", font=("Helvetica", 15), justification='center')],
        [sg.Image(filename='', key='-IMAGE-')],
        [sg.Text("Name of PPL", font=("Helvetica", 18), justification='center', key='-NAME_LABEL-')],
        [sg.Text("Confidence: ", font=("Helvetica", 18), justification='center', key='-CONFIDENCE_LABEL-')],
        # [sg.Text("Time: ", font=("Helvetica", 18), justification='center')],
        # [sg.Text("Date: ", font=("Helvetica", 18), justification='center')],
        [sg.Checkbox("Detect Image: ", default=True, key='-DETECT_FACE-')],
    ]

    menu_column = [
        [sg.Text("Name: ", font=("Helvetica", 15), justification='center'), sg.InputText(key='-NAME_TEXTBOX-')],
        [sg.HorizontalSeparator()],
        [sg.Text("Add Photos..", font=("Helvetica", 15), justification='center')],
        [sg.Button('Using Camera Capture', key='-CAMERA_IMPORT-'),
            sg.Input(key='-FILE_IMPORT-',visible=False,enable_events=True),sg.FilesBrowse('Using File Import', target='-FILE_IMPORT-', file_types=(('Image Files', '*.png *.jpg *.jpeg'),))],
        [sg.Column(layout = [], key='-IMAGE_COLUMN-',scrollable=True, vertical_scroll_only=True, s= (450, 500))],
        [sg.Button('Clear Selected Photos', key='-CLEAR_PHOTOS-')],
        [sg.Button('Save and Compile', key='-COMPILE_PHOTOS-')],
        [sg.HorizontalSeparator()],
        [sg.Button('Exit')]
    ]
    # Create the PySimpleGUI window layout
    main_layout = [
        [
            sg.Col(image_column),
            sg.VSeperator(),
            sg.Col(menu_column),
        ]
    ]
    return main_layout

window = sg.Window('Face Detection and Recognition', updateLayout(), finalize=True)

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

while True:
    event, values = window.read(timeout=20)  # Read events and refresh window every 20 milliseconds
    ret, frame = cap.read()  # Capture a frame from the webcam
    if event == sg.WIN_CLOSED or event == 'Exit':
        shutil.rmtree('temp', ignore_errors=True)
        break
    elif event == '-CAMERA_IMPORT-':
        Path(f"temp/").mkdir(exist_ok=True)
        cwd = os.getcwd()
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]
        cv2.imwrite(f'{cwd}\\temp\\{current_time}.png', frame)
        selected_files.append(f'{cwd}\\temp\\{current_time}.png')
        window.extend_layout(window['-IMAGE_COLUMN-'], [[sg.Image(filename=selected_files[-1], size=(400, 400))]])
        window.visibility_changed()
        window['-IMAGE_COLUMN-'].contents_changed()
    elif event == '-FILE_IMPORT-':
        new_files = values['-FILE_IMPORT-'].split(';')
        selected_files += new_files
        # window.extend_layout(window['-IMAGE_COLUMN-'], [[sg.T('A New Input Line'), sg.I(key=f'-IN-TEST-')]])
        new_pics = []
        for file in new_files:
            # print(file)
            if file.split('.')[-1] == "jpg":
                new_pics.append(sg.Image(filename="assets\catch_jpg.png", size=(400, 400)))
            else:
                new_pics.append(sg.Image(filename=file, size=(400, 400)))
        # new_pics = [sg.Image(filename=file, size=(400, 400)) for file in new_files]
        print(new_pics)
        for pic in new_pics:
            window.extend_layout(window['-IMAGE_COLUMN-'], [[pic]])
        
        window.visibility_changed()
        window['-IMAGE_COLUMN-'].contents_changed()
    elif event == '-CLEAR_PHOTOS-':
        selected_files = []
        shutil.rmtree('temp', ignore_errors=True)
        window.close()
        window = sg.Window('Face Detection and Recognition', updateLayout(), finalize=True)
    elif event == '-COMPILE_PHOTOS-':
        name = values['-NAME_TEXTBOX-'].strip()
        # if name == '':
        #     sg.popup_error('Please Enter a Name')
        #     continue

        images_dir = Path(f"images/{name}")
        images_dir.mkdir(exist_ok=True)
        
        for file in selected_files:
            file = Path(file)
            os.rename(file, f'{images_dir}/{file.name}')

        shutil.rmtree('temp', ignore_errors=True)
        shutil.rmtree('training_data',ignore_errors=True)
        shutil.rmtree('__pycache__',ignore_errors=True)
        selected_files = []

        train_model()

        window.close()
        window = sg.Window('Face Detection and Recognition', updateLayout(), finalize=True)

    if ret:
        if values['-DETECT_FACE-']:
            frame, name, conf = detect_faces(frame)

            if window['-NAME_LABEL-'] != '':
                window['-NAME_LABEL-'].update(name)
                pass
            else:
                window['-NAME_LABEL-'].update("No Face Detected")
            # frame = frame1
            
            window['-CONFIDENCE_LABEL-'].update(conf)

        frame_resized = cv2.resize(frame, (640, 480))
        img_bytes = cv2.imencode('.png', frame_resized)[1].tobytes()
        window['-IMAGE-'].update(data=img_bytes)

    

cap.release()  # Release the webcam
window.close()  # Close the PySimpleGUI window