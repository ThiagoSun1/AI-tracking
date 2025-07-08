#!/usr/bin/env python3
"""
Soccer Ball Tracker with Servo Control and Clean Video Recording
Tracks soccer ball and uses servos to physically follow it while recording clean video
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
from datetime import datetime
try:
    import board
    import busio
    from adafruit_pca9685 import PCA9685
    from adafruit_motor import servo
    SERVO_AVAILABLE = True
except ImportError:
    print("Warning: Adafruit libraries not available. Servo control disabled.")
    print("Install with: pip install adafruit-circuitpython-pca9685 adafruit-circuitpython-motor")
    SERVO_AVAILABLE = False

class ServoController:
    def __init__(self, pan_channel=0, tilt_channel=1, i2c_address=0x40):
        """
        Initialize Adafruit servo controller
        
        Args:
            pan_channel (int): PCA9685 channel for pan servo (0-15)
            tilt_channel (int): PCA9685 channel for tilt servo (0-15)
            i2c_address (hex): I2C address of PCA9685 board (default 0x40)
        """
        if not SERVO_AVAILABLE:
            print("Servo control not available - running in simulation mode")
            self.enabled = False
            return
            
        self.pan_channel = pan_channel
        self.tilt_channel = tilt_channel
        self.enabled = True
        
        try:
            # Initialize I2C and PCA9685
            self.i2c = busio.I2C(board.SCL, board.SDA)
            self.pca = PCA9685(self.i2c, address=i2c_address)
            self.pca.frequency = 50  # 50Hz for servos
            
            # Create servo objects
            self.pan_servo = servo.Servo(self.pca.channels[pan_channel])
            self.tilt_servo = servo.Servo(self.pca.channels[tilt_channel])
            
            # Set servo ranges (adjust these based on your specific servos)
            self.pan_servo.set_pulse_width_range(min_pulse=500, max_pulse=2500)
            self.tilt_servo.set_pulse_width_range(min_pulse=500, max_pulse=2500)
            
            print(f"Adafruit PCA9685 servo controller initialized")
            print(f"Pan servo: Channel {pan_channel}, Tilt servo: Channel {tilt_channel}")
            
        except Exception as e:
            print(f"Error initializing Adafruit servo controller: {e}")
            self.enabled = False
            return
        
        # Servo limits (in degrees)
        self.pan_min_deg = -90
        self.pan_max_deg = 90
        self.tilt_min_deg = -45
        self.tilt_max_deg = 45
        
        # Current positions
        self.current_pan = 0
        self.current_tilt = 0
        
        # Movement constraints
        self.max_speed = 5.0  # degrees per frame
        self.dead_zone = 20   # pixels - don't move if ball is within this range of center
        
        # Move to center position
        self.move_to_position(0, 0)
        print(f"Servos initialized and centered")
    
    def move_to_position(self, pan_degrees, tilt_degrees):
        """Move servos to specific position using Adafruit library"""
        if not self.enabled:
            return
            
        # Clamp to limits
        pan_degrees = max(self.pan_min_deg, min(self.pan_max_deg, pan_degrees))
        tilt_degrees = max(self.tilt_min_deg, min(self.tilt_max_deg, tilt_degrees))
        
        self.current_pan = pan_degrees
        self.current_tilt = tilt_degrees
        
        try:
            # Move servos using Adafruit library (handles angle conversion automatically)
            self.pan_servo.angle = pan_degrees + 90   # Convert to 0-180 range
            self.tilt_servo.angle = tilt_degrees + 90  # Convert to 0-180 range
        except Exception as e:
            print(f"Error moving servos: {e}")
            self.enabled = False
    
    def track_target(self, target_x, target_y, frame_width, frame_height):
        """
        Calculate servo movements to track target
        
        Args:
            target_x, target_y: Target position in pixels
            frame_width, frame_height: Frame dimensions
        """
        if not self.enabled:
            return
            
        # Calculate center offset
        center_x = frame_width // 2
        center_y = frame_height // 2
        
        error_x = target_x - center_x
        error_y = target_y - center_y
        
        # Check if within dead zone
        if abs(error_x) < self.dead_zone and abs(error_y) < self.dead_zone:
            return
        
        # Calculate required movement (proportional control)
        # Scale pixel error to degree movement
        pan_movement = -(error_x / frame_width) * 45  # Negative for correct direction
        tilt_movement = (error_y / frame_height) * 30
        
        # Limit movement speed
        pan_movement = max(-self.max_speed, min(self.max_speed, pan_movement))
        tilt_movement = max(-self.max_speed, min(self.max_speed, tilt_movement))
        
        # Calculate new positions
        new_pan = self.current_pan + pan_movement
        new_tilt = self.current_tilt + tilt_movement
        
        # Move servos
        self.move_to_position(new_pan, new_tilt)
    
    def center_servos(self):
        """Move servos to center position"""
        self.move_to_position(0, 0)
    
    def cleanup(self):
        """Clean up Adafruit servo resources"""
        if not self.enabled:
            return
            
        try:
            # Center servos before shutdown
            self.center_servos()
            time.sleep(0.5)
            
            # Deinitialize PCA9685
            self.pca.deinit()
            print("Adafruit servo controller cleaned up")
        except Exception as e:
            print(f"Error during servo cleanup: {e}")

class SoccerBallTrackerWithServos:
    def __init__(self, model_path, camera_index=0, output_dir=None, 
                 pan_channel=0, tilt_channel=1, enable_servos=True, i2c_address=0x40):
        """
        Initialize the soccer ball tracker with Adafruit servo control
        
        Args:
            model_path (str): Path to your trained YOLO model
            camera_index (int): Camera index (usually 0 for default camera)
            output_dir (str): Directory to save videos (default: Desktop)
            pan_channel (int): PCA9685 channel for pan servo (0-15)
            tilt_channel (int): PCA9685 channel for tilt servo (0-15)
            enable_servos (bool): Enable servo control
            i2c_address (hex): I2C address of PCA9685 board
        """
        # Set output directory (Desktop by default)
        if output_dir is None:
            home_dir = os.path.expanduser("~")
            self.output_dir = os.path.join(home_dir, "Desktop")
        else:
            self.output_dir = output_dir
            
        # Make sure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load YOLO model
        print(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)
        
        # Initialize camera
        print(f"Initializing camera {camera_index}")
        self.cap = cv2.VideoCapture(camera_index)
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Check if camera opened successfully
        if not self.cap.isOpened():
            raise Exception("Error: Could not open camera")
            
        # Get actual camera properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        print(f"Camera initialized: {self.width}x{self.height} @ {self.fps}fps")
        
        # Initialize servo controller
        self.servo_enabled = enable_servos
        if self.servo_enabled:
            self.servo_controller = ServoController(pan_channel, tilt_channel, i2c_address)
            if not self.servo_controller.enabled:
                print("Servo control disabled - continuing without servo tracking")
                self.servo_enabled = False
        else:
            print("Servo control disabled by user")
            self.servo_controller = None
        
        # Tracking variables
        self.ball_positions = []  # Store last positions for trajectory
        self.max_positions = 30   # Max positions to store
        self.last_detection_time = 0
        self.detection_timeout = 2.0  # seconds
        
        # Colors (BGR format) - only for display
        self.bbox_color = (0, 255, 0)      # Green for bounding box
        self.center_color = (0, 0, 255)    # Red for center point
        self.trail_color = (255, 0, 0)     # Blue for trajectory trail
        self.servo_color = (255, 255, 0)   # Cyan for servo info
        
        # Video recording variables
        self.video_writer = None
        self.recording = False
        self.output_filename = None
        self.fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        
        # Tracking state
        self.tracking_active = False
        self.last_target_x = self.width // 2
        self.last_target_y = self.height // 2
        
    def start_recording(self):
        """Start recording clean video (no bounding boxes)"""
        if self.recording:
            print("Already recording!")
            return
            
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_filename = os.path.join(self.output_dir, f"soccer_tracking_{timestamp}.avi")
        
        # Initialize video writer
        self.video_writer = cv2.VideoWriter(
            self.output_filename,
            self.fourcc,
            self.fps,
            (self.width, self.height)
        )
        
        if not self.video_writer.isOpened():
            print("Error: Could not open video writer")
            return
            
        self.recording = True
        print(f"Started recording clean video: {self.output_filename}")
        
    def stop_recording(self):
        """Stop recording video"""
        if not self.recording:
            print("Not currently recording!")
            return
            
        self.recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            
        print(f"Recording stopped. Clean video saved: {self.output_filename}")
        
    def draw_trajectory(self, frame):
        """Draw trajectory trail of the soccer ball (display only)"""
        if len(self.ball_positions) > 1:
            # Draw lines connecting previous positions
            for i in range(1, len(self.ball_positions)):
                cv2.line(frame, 
                        self.ball_positions[i-1], 
                        self.ball_positions[i], 
                        self.trail_color, 2)
    
    def draw_servo_info(self, frame):
        """Draw servo information on display frame"""
        if not self.servo_enabled or not self.servo_controller:
            cv2.putText(frame, "Servo: DISABLED", (10, 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return
            
        # Draw servo positions
        pan_pos = self.servo_controller.current_pan
        tilt_pos = self.servo_controller.current_tilt
        
        cv2.putText(frame, f"Pan: {pan_pos:.1f}°", (10, 60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.servo_color, 2)
        cv2.putText(frame, f"Tilt: {tilt_pos:.1f}°", (10, 85), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.servo_color, 2)
        
        # Draw tracking status
        status = "TRACKING" if self.tracking_active else "SEARCHING"
        color = (0, 255, 0) if self.tracking_active else (0, 165, 255)
        cv2.putText(frame, f"Status: {status}", (10, 110), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw center crosshairs
        center_x, center_y = self.width // 2, self.height // 2
        cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (255, 255, 255), 1)
        cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (255, 255, 255), 1)
        
        # Draw dead zone
        dead_zone = self.servo_controller.dead_zone if self.servo_controller else 20
        cv2.rectangle(frame, 
                     (center_x - dead_zone, center_y - dead_zone),
                     (center_x + dead_zone, center_y + dead_zone),
                     (128, 128, 128), 1)
    
    def process_frame_for_display(self, frame):
        """
        Process frame for display with tracking overlays and servo control
        
        Args:
            frame: Input frame from camera
            
        Returns:
            display_frame: Frame with tracking overlays for display
        """
        display_frame = frame.copy()
        current_time = time.time()
        ball_detected = False
        
        # Run YOLO inference
        results = self.model(frame, conf=0.5, verbose=False)
        
        # Process detections
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Calculate center point
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    # Update tracking state
                    ball_detected = True
                    self.last_detection_time = current_time
                    self.last_target_x = center_x
                    self.last_target_y = center_y
                    
                    # Control servos to track the ball
                    if self.servo_enabled and self.servo_controller:
                        self.servo_controller.track_target(center_x, center_y, 
                                                         self.width, self.height)
                    
                    # Add to trajectory
                    self.ball_positions.append((center_x, center_y))
                    if len(self.ball_positions) > self.max_positions:
                        self.ball_positions.pop(0)
                    
                    # Draw bounding box on display frame only
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), self.bbox_color, 2)
                    
                    # Draw center point on display frame only
                    cv2.circle(display_frame, (center_x, center_y), 5, self.center_color, -1)
                    
                    # Draw confidence and class label on display frame only
                    label = f"Soccer Ball: {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(display_frame, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), self.bbox_color, -1)
                    cv2.putText(display_frame, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    # Display coordinates on display frame only
                    coord_text = f"({center_x}, {center_y})"
                    cv2.putText(display_frame, coord_text, (center_x + 10, center_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.center_color, 1)
        
        # Update tracking status
        self.tracking_active = ball_detected or (current_time - self.last_detection_time < self.detection_timeout)
        
        # If no ball detected for too long, maybe center the servos or do a search pattern
        if not self.tracking_active and self.servo_enabled and self.servo_controller:
            # Could implement a search pattern here
            pass
        
        # Draw trajectory trail on display frame only
        self.draw_trajectory(display_frame)
        
        # Draw servo information
        self.draw_servo_info(display_frame)
        
        return display_frame
    
    def run(self):
        """Main tracking and recording loop"""
        print("Starting soccer ball tracking with servo control...")
        print("Controls:")
        print("  SPACE - Start/Stop recording")
        print("  'q' - Quit")
        print("  'r' - Reset trajectory")
        print("  's' - Save screenshot")
        print("  'c' - Center servos")
        print("  't' - Toggle servo tracking")
        
        # FPS calculation variables
        fps_counter = 0
        start_time = time.time()
        
        try:
            while True:
                # Read frame from camera
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Record CLEAN frame (original, no overlays) if recording
                if self.recording and self.video_writer:
                    self.video_writer.write(frame)  # Write original clean frame
                
                # Process frame for display with tracking overlays
                display_frame = self.process_frame_for_display(frame)
                
                # Calculate and display FPS on display frame only
                fps_counter += 1
                elapsed_time = time.time() - start_time
                if elapsed_time >= 1.0:
                    fps = fps_counter / elapsed_time
                    fps_counter = 0
                    start_time = time.time()
                    
                    # Display FPS on display frame only
                    cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add recording indicator to display frame only
                if self.recording:
                    cv2.circle(display_frame, (self.width - 30, 30), 10, (0, 0, 255), -1)  # Red dot
                    cv2.putText(display_frame, "REC", (self.width - 60, 35), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(display_frame, "Press SPACE to record", (self.width - 200, 35), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display frame info on display frame only
                cv2.putText(display_frame, f"Resolution: {self.width}x{self.height}", 
                          (10, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show the display frame (with overlays)
                cv2.imshow('Soccer Ball Tracker with Servos', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):  # Spacebar - start/stop recording
                    if self.recording:
                        self.stop_recording()
                    else:
                        self.start_recording()
                elif key == ord('r'):
                    # Reset trajectory
                    self.ball_positions = []
                    print("Trajectory reset")
                elif key == ord('s'):
                    # Save clean screenshot (no overlays)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    screenshot_filename = os.path.join(self.output_dir, f"clean_screenshot_{timestamp}.jpg")
                    cv2.imwrite(screenshot_filename, frame)  # Save original clean frame
                    print(f"Clean screenshot saved: {screenshot_filename}")
                elif key == ord('c'):
                    # Center servos
                    if self.servo_enabled and self.servo_controller:
                        self.servo_controller.center_servos()
                        print("Servos centered")
                elif key == ord('t'):
                    # Toggle servo tracking
                    self.servo_enabled = not self.servo_enabled
                    status = "enabled" if self.servo_enabled else "disabled"
                    print(f"Servo tracking {status}")
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
    def cleanup(self):
        """Clean up resources"""
        if self.recording:
            self.stop_recording()
            
        if self.servo_controller:
            self.servo_controller.cleanup()
            
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Cleanup completed")

def main():
    # Configuration
    MODEL_PATH = "runs/detect/train3/weights/best.pt"  # Update this path to your model
    CAMERA_INDEX = 0  # Usually 0 for default camera, try 1, 2 if needed
    OUTPUT_DIR = None  # None = Desktop, or specify custom path
    
    # Adafruit PCA9685 servo configuration
    PAN_SERVO_CHANNEL = 0    # PCA9685 channel for pan servo (0-15)
    TILT_SERVO_CHANNEL = 1   # PCA9685 channel for tilt servo (0-15)
    I2C_ADDRESS = 0x40       # I2C address of PCA9685 board (default 0x40)
    ENABLE_SERVOS = True     # Set to False to disable servo control
    
    try:
        # Create and run tracker with Adafruit servo control
        tracker = SoccerBallTrackerWithServos(
            MODEL_PATH, 
            CAMERA_INDEX, 
            OUTPUT_DIR,
            PAN_SERVO_CHANNEL,
            TILT_SERVO_CHANNEL,
            ENABLE_SERVOS,
            I2C_ADDRESS
        )
        tracker.run()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up
        if 'tracker' in locals():
            tracker.cleanup()

if __name__ == "__main__":
    main()
