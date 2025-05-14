# Automated-Defense-System
This project is a real-time smart security system that uses facial recognition, servo motors, and Raspberry Pi to detect, track, and respond to unknown individuals entering a secured area. It is designed to simulate an automated defense mechanism for restricted zones.
Key Features
✅ Real-time facial detection and recognition using OpenCV

🔊 Buzzer warning system on detecting unknown faces

🔁 Servo motor control to track facial movement

🛑 Manual override button – fires only if not pressed within 3 seconds

🎯 Simulated target locking and firing mechanism

🛠️ Tech Stack
Python 3

OpenCV

Raspberry Pi 4 (4GB)

Servo Motors (SG90)

GPIO with RPi.GPIO / gpiozero

Buzzer and Push Button

🧠 Functional Flow
Camera captures live video stream.

Face is detected and checked against known faces.

If unknown:

A buzzer is triggered.

User is prompted to press a button within 3 seconds.

If no input, servo motor locks onto the face and simulates firing.

If recognized:

Displays Name and authorised
