import sqlite3
import cv2
import os
from flask import Flask, request, render_template, redirect, session, url_for , send_file
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import time
import matplotlib
matplotlib.use('TkAgg') 




# VARIABLES
MESSAGE = "WELCOME  " \
          " Instruction: If you are a new, please register by entering your details"

# Defining Flask App
app = Flask(__name__, static_url_path='/static')

app.secret_key = 'your_secret_key_here'  


# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
formatted_datetoday = date.today().strftime("%d-%B-%Y")


# Initializing VideoCapture object to access Webcam
face_detector = cv2.CascadeClassifier('HARS.xml')

try:
    cap = cv2.VideoCapture(0)  # Try the default camera (index 0)
except Exception as e:
    print(f"Error: {e}")
    # Handle the error, show a message to the user, or fall back to a different camera index if available.


# If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

# Get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))

# Extract the face from an image
def extract_faces(img):
    if img is not None and len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.3, 5)
        return face_points
    else:
        return []

# Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

# A function that trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')



# Extract info from today's attendance file in the attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S %d/%m/%Y')
    return df['Name'], df['Roll'], df['Time'], len(df)


# Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]

    # Get the current date and time
    current_datetime = datetime.now().strftime("%H:%M:%S %d/%m/%Y")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if str(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_datetime}')
    else:
        print("This user has already marked attendance for the day, but still, I am marking it.")


# ROUTING FUNCTIONS

# Our main page
@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                           datetoday2=formatted_datetoday, mess=MESSAGE)

# This function will run when we click on Take Attendance Button
@app.route('/start', methods=['GET'])
def start():
    ATTENDANCE_MARKED = False
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        names, rolls, times, l = extract_attendance()
        MESSAGE = 'This face is not registered with us, kindly register yourself first'
        print("Face not in the database, need to register")
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                               datetoday2=formatted_datetoday, mess=MESSAGE)

    cap = cv2.VideoCapture(0)
    ret = True
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            cv2.putText(frame, f'{identified_person}', (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2)
            
            if cv2.waitKey(25) == ord('a'):
                add_attendance(identified_person)
                current_time_ = datetime.now().strftime("%H:%M:%S")
                print(f"Attendance marked for {identified_person}, at {current_time_}")
                ATTENDANCE_MARKED = True
                break
        if ATTENDANCE_MARKED:
            break
        

        # Display the resulting frame
        cv2.imshow('Attendance Check, press "a" to exit', frame)
        cv2.putText(frame, 'hello', (30, 30), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255))

      
        if cv2.waitKey(25) == ord('a'):
            add_attendance(identified_person)
            current_time_ = datetime.now().strftime("%H:%M:%S")
            print(f"Attendance marked for {identified_person}, at {current_time_}")
            ATTENDANCE_MARKED = True
            break

    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    MESSAGE = 'Attendance taken successfully'
    print("Attendance registered")
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                           datetoday2=formatted_datetoday, mess=MESSAGE)

@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/' + newusername + '_' + str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cap = cv2.VideoCapture(0)
    i, j = 0, 0
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/50', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,
                        cv2.LINE_AA)
            if j % 10 == 0:
                name = newusername + '_' + str(i) + '.jpg'
                cv2.imwrite(userimagefolder + '/' + name, frame[y:y + h, x:x + w])
                i += 1
            j += 1
        if j == 500:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names, rolls, times, l = extract_attendance()
    if totalreg() > 0:
        names, rolls, times, l = extract_attendance()
        MESSAGE = 'User added Successfully'
        print("Message changed")
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                               datetoday2=formatted_datetoday, mess=MESSAGE)
    else:
        return redirect(url_for('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                                datetoday2=formatted_datetoday))

# Admin login route
@app.route('/adminlogin', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'admin123':  # Replace with your admin credentials
            session['admin_logged_in'] = True
            return redirect('/admin_dashboard')
        else:
            return "Invalid username or password"

    # Add this line to handle GET requests
    return render_template('adminlogin.html')


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Define the function
def formatted_datetoday():
    return date.today().strftime("%d-%B-%Y")

# Admin dashboard route
@app.route('/admin_dashboard')
def admin_dashboard():
    global chart_path  # Declare chart_path as global

    if session.get('admin_logged_in'):
        # Admin is logged in, fetch attendance data
        names, rolls, times, l = extract_attendance()

        # Calculate total attendance for the current month
        current_month = datetime.now().month
        total_attendance = 0

        for i in range(len(names)):
            attendance_date = datetime.strptime(times[i].strftime('%H:%M:%S'), '%H:%M:%S')
            if attendance_date.month == current_month:
                total_attendance += 1


        # Calculate the attendance counts for each student
        attendance_counts = names.value_counts()

        # Check if attendance_counts is empty
        if attendance_counts.empty:
            # Handle the case when attendance_counts is empty
            error_message = "No attendance data available."
            
            # Pass the error message to the template
            return render_template('admindashboard.html', error_message=error_message)


        # Plot the attendance counts for each student
        plt.figure(figsize=(12, 6))
        attendance_counts.plot(kind='bar')
        plt.title('Attendance Counts for Each Student')
        plt.xlabel('Student Name')
        plt.ylabel('Count')

        # Save the chart as an image
        chart_path = 'static/chart.png'
        plt.savefig(chart_path)
        plt.close()

        # Calculate attendance percentages for each student
        overall_count = 30
        attendance_percentages = (attendance_counts / overall_count) * 100

        # Plot the attendance percentages for each student
        plt.figure(figsize=(10, 6))
        plt.bar(attendance_percentages.index, attendance_percentages.values)
        plt.title('Attendance Percentage for Each Student')
        plt.xlabel('Student Name')
        plt.ylabel('Attendance Percentage')
        plt.ylim(0, 100)  # Set y-axis limit to 100%
        plt.savefig('static/attendance_percentage_chart.png')
        plt.close()

        # Pass the data to the template
        return render_template('admindashboard.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                               datetoday2=formatted_datetoday(), total_attendance=total_attendance,
                               chart_path=chart_path, attendance_counts=attendance_counts)

    else:
        return redirect('/adminlogin')





# Logout route
@app.route('/logout')
def logout():
    session.pop('admin_logged_in', None)
    return redirect('/adminlogin')




    
# Delete attendance route
@app.route('/delete_attendance', methods=['POST'])
def delete_attendance():
    if request.method == 'POST':
        index = int(request.form.get('index'))  
        delete_attendance_by_index(index)
    return redirect('/')
def delete_attendance_by_index(index):
    # Load the attendance data
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    
    # Check if the index is within the valid range
    if 0 <= index < len(df):
        # Delete the record at the specified index
        df = df.drop(index, axis=0)
        
        # Save the updated attendance data
        df.to_csv(f'Attendance/Attendance-{datetoday}.csv', index=False)


# Attendancereport route
@app.route('/attendancereport')
def attendancereport():
    # Extract attendance data
    names, rolls, times, l = extract_attendance()

    # Calculate total attendance for the current month
    current_month = datetime.now().month
    total_attendance = 0

    for i in range(len(names)):
        attendance_date = datetime.strptime(times[i].strftime('%H:%M:%S'), '%H:%M:%S')
        if attendance_date.month == current_month:
            total_attendance += 1

    # Path to the CSV file
    csv_path = f'Attendance/Attendance-{datetoday}.csv'

    # Pass data and file path to the template
    return render_template('attendancereport.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                           datetoday2=formatted_datetoday(), total_attendance=total_attendance, csv_path=csv_path)



# Add this route for exporting and downloading the CSV file
@app.route('/export_attendance', methods=['GET'])
def export_attendance():
    # Path to the CSV file
    csv_path = f'Attendance/Attendance-{datetoday}.csv'

    # Set up response headers to trigger a download
    response = send_file(csv_path, as_attachment=True)
    response.headers["Content-Disposition"] = f"attachment; filename=Attendance-{datetoday}.csv"
    response.headers["Content-Type"] = "text/csv"

    # Return the file as a response
    return response

# Main function to run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=100)
