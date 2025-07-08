# AI Soccer Ball Tracker

A Jetson Nano-based object detection system that **tracks a soccer ball using AI in real time** and records the game or training session using a Logitech C270 webcam and servo-controlled panning Built entirely in Python, it uses two MG995 servo motors and a PCA9685 controller to physically follow the ball across the field.

---

##  ⚽ Overview

This project captures video and automatically follows the **soccer ball or whatever yo want to track** during games or training sessions. The system identifies the soccer ball using object detection and moves the camera using servo motors to keep it in frame. Recordings are saved for later analysis.

---

## 🧠 Features

- 🎯 Tracks a color of your choice in real time using OpenCV
- 📷 Records the a soccer match or training session with the Logitech C270 webcam
- 🔄 Dual MG995 servo motors (pan & tilt) controlled by PCA9685
- ⚙️ Runs entirely on a Jetson Nano
- 🐍 Written in pure Python with no external apps or software needed

---

## 📦 Hardware Requirements

- Jetson Nano Jetpack 4.6.4
- Logitech C270 webcam
- 2× MG995 servo motors
- PCA9685 16-channel PWM servo driver
- External 5–6V power supply for servos
- Tripod or camera mount (optional but useful)

---

## 🧰 Software Requirements

Before installing anything, make sure you're in a virtual enviroment:

```bash
# install package to create virtual enviroment
pip install virtualenv

# make virtual enviroment
virtualenv venv

# activate virtual enviroment
source venv/bin/activate

```

- Python 3.8 (pre-installed on Jetson Nano)
- The following Python libraries:

```bash
sudo apt update
sudo apt install python3-pip

# install necessary dependencies
pip install -r requirements.txt --force-reinstall

#check if pca9685 is connected correctly; it should show a 40 in the 40 row and the 0 column
sudo i2cdetect -y -r 1

# clone the repository
git clone https://github.com/ThiagoSun1/AI-tracking
cd ~/AI-tracking

# run the tracker
python3 yolo.py

```

## 📄 License

This project is licensed under the [MIT License](LICENSE).
