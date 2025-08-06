#!/usr/bin/python3
import cv2
import numpy as np
import time
from ultralytics import YOLO
from adafruit_servokit import ServoKit

class PIDController:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()
    
    def update(self, error):
        current_time = time.time()
        dt = current_time - self.last_time
        
        if dt <= 0:
            dt = 0.01
        
        # PID calculation
        p_term = self.kp * error
        self.integral += error * dt
        i_term = self.ki * self.integral
        derivative = (error - self.prev_error) / dt
        d_term = self.kd * derivative
        
        output = p_term + i_term + d_term
        
        # Update for next iteration
        self.prev_error = error
        self.last_time = current_time
        
        return max(-3, min(3, output))  # Limit output for smooth movement

class BallTracker:
    def __init__(self, model_path, target_class='sports ball', camera_device='/dev/video0'):
        print("Initializing Ball Tracker...")
        
        # Initialize YOLO model
        self.model = YOLO(model_path)
        self.target_class = target_class
        
        # Initialize webcam with specific device
        self.camera = None
        self.camera_device = camera_device
        
        # Try to open the specific camera device
        try:
            print(f"Attempting to open camera at {camera_device}...")
            
            # First try with V4L2 backend (preferred for Linux)
            self.camera = cv2.VideoCapture(0, cv2.CAP_V4L2)  # /dev/video0 maps to device 0
            
            if not self.camera.isOpened():
                print("V4L2 backend failed, trying default backend...")
                self.camera.release()
                self.camera = cv2.VideoCapture(0)  # Try with default backend
            
            if not self.camera.isOpened():
                raise RuntimeError(f"Cannot open camera device {camera_device}")
            
            # Test if we can actually read frames
            ret, test_frame = self.camera.read()
            if not ret or test_frame is None:
                raise RuntimeError(f"Cannot read frames from {camera_device}")
            
            print(f"Successfully opened camera: {camera_device}")
            
        except Exception as e:
            if self.camera:
                self.camera.release()
            raise RuntimeError(f"Failed to initialize camera {camera_device}: {e}")
        
        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        # Set additional properties for better performance
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize latency
        
        # Verify settings
        actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
        print(f"Camera settings: {actual_width}x{actual_height} @ {actual_fps} FPS")
        
        # Initialize ServoKit
        self.kit = ServoKit(channels=16)
        self.pan_pos = 90.0
        self.tilt_pos = 90.0
        
        # Center servos
        self.kit.servo[0].angle = self.pan_pos
        self.kit.servo[1].angle = self.tilt_pos
        time.sleep(2)  # Give servos time to move
        print("Servos centered - Pan: channel 0, Tilt: channel 1")
        
        # PID controllers (tuned for smooth tracking)
        self.pan_pid = PIDController(kp=0.6, ki=0.02, kd=0.1)
        self.tilt_pid = PIDController(kp=0.6, ki=0.02, kd=0.1)
        
        # Frame parameters
        self.frame_width = 640
        self.frame_height = 480
        self.frame_center_x = self.frame_width // 2
        self.frame_center_y = self.frame_height // 2
        
        # Tracking parameters
        self.tracking_active = False
        self.last_detection_time = 0
        self.detection_timeout = 3.0
        self.movement_threshold = 15  # Minimum pixel error to move servos
        
        # Servo limits
        self.pan_min, self.pan_max = 0, 180
        self.tilt_min, self.tilt_max = 30, 150
        
        print("Ball Tracker ready!")
        
    def detect_ball(self, frame):
        """Detect ball using YOLO and return best detection"""
        results = self.model(frame, conf=0.25, device='cpu', verbose=False)
        
        best_detection = None
        best_confidence = 0
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf[0])
                    
                    # Check if it's our target class and has good confidence
                    if (self.target_class.lower() in class_name.lower() and 
                        confidence > best_confidence and confidence > 0.3):
                        
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        
                        best_detection = {
                            'center_x': center_x,
                            'center_y': center_y,
                            'confidence': confidence,
                            'bbox': (int(x1), int(y1), int(x2), int(y2))
                        }
                        best_confidence = confidence
        
        if best_detection:
            return (best_detection['center_x'], best_detection['center_y'], 
                   best_detection['confidence'], best_detection['bbox'])
        
        return None, None, None, None
    
    def move_servos(self, pan_angle, tilt_angle):
        """Move servos with safety limits"""
        # Apply limits
        pan_angle = max(self.pan_min, min(self.pan_max, pan_angle))
        tilt_angle = max(self.tilt_min, min(self.tilt_max, tilt_angle))
        
        try:
            self.kit.servo[0].angle = pan_angle
            self.kit.servo[1].angle = tilt_angle
            self.pan_pos = pan_angle
            self.tilt_pos = tilt_angle
        except Exception as e:
            print(f"Servo error: {e}")
    
    def search_pattern(self):
        """Simple search pattern when ball is lost"""
        current_time = time.time()
        # Slow sweep pattern
        sweep_angle = 25 * np.sin(current_time * 0.5)  # 25 degree sweep
        search_pan = 90 + sweep_angle
        self.move_servos(search_pan, self.tilt_pos)
    
    def track_ball(self):
        """Main tracking loop"""
        print("\n=== BALL TRACKING STARTED ===")
        print(f"Camera device: {self.camera_device}")
        print("Controls:")
        print("  'q' - Quit")
        print("  'c' - Center servos")
        print("  's' - Toggle search mode")
        print("  'r' - Reset camera connection")
        print("================================")
        
        self.tracking_active = True
        search_mode = False
        frame_count = 0
        
        try:
            while self.tracking_active:
                # Capture frame from webcam
                ret, frame_bgr = self.camera.read()
                if not ret:
                    print("Failed to capture frame from webcam")
                    # Try to reconnect camera
                    print("Attempting to reconnect camera...")
                    self.camera.release()
                    time.sleep(1)
                    self.camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
                    if not self.camera.isOpened():
                        self.camera = cv2.VideoCapture(0)
                    continue
                
                # Run detection every few frames for better performance
                if frame_count % 1 == 0:
                    ball_x, ball_y, confidence, bbox = self.detect_ball(frame_bgr)
                    
                    if ball_x is not None:
                        self.last_detection_time = time.time()
                        search_mode = False
                        
                        # Calculate tracking errors
                        pan_error = self.frame_center_x - ball_x
                        tilt_error = ball_y - self.frame_center_y  # Inverted for camera
                        
                        # Only move if error is significant (reduces jitter)
                        if abs(pan_error) > self.movement_threshold or abs(tilt_error) > self.movement_threshold:
                            # Get PID outputs
                            pan_output = self.pan_pid.update(pan_error)
                            tilt_output = self.tilt_pid.update(tilt_error)
                            
                            # Move servos
                            new_pan = self.pan_pos - pan_output
                            new_tilt = self.tilt_pos + tilt_output
                            self.move_servos(new_pan, new_tilt)
                        
                        # Draw tracking visualization
                        cv2.circle(frame_bgr, (ball_x, ball_y), 12, (0, 255, 0), 3)
                        cv2.rectangle(frame_bgr, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                        cv2.putText(frame_bgr, f'TRACKING: {confidence:.2f}', 
                                   (bbox[0], bbox[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # Draw line from center to ball
                        cv2.line(frame_bgr, (self.frame_center_x, self.frame_center_y), 
                                (ball_x, ball_y), (0, 255, 255), 2)
                        
                        print(f"Tracking: Ball at ({ball_x:3d}, {ball_y:3d}) | "
                              f"Pan: {self.pan_pos:5.1f}° | Tilt: {self.tilt_pos:5.1f}° | "
                              f"Conf: {confidence:.2f}")
                    
                    else:
                        # No ball detected
                        time_since_detection = time.time() - self.last_detection_time
                        
                        if time_since_detection > self.detection_timeout:
                            if not search_mode:
                                print("Ball lost - Starting search pattern")
                                search_mode = True
                            self.search_pattern()
                
                # Draw UI elements
                # Crosshairs at center
                cv2.line(frame_bgr, (self.frame_center_x-25, self.frame_center_y), 
                        (self.frame_center_x+25, self.frame_center_y), (255, 0, 0), 2)
                cv2.line(frame_bgr, (self.frame_center_x, self.frame_center_y-25), 
                        (self.frame_center_x, self.frame_center_y+25), (255, 0, 0), 2)
                
                # Status display
                status_color = (0, 255, 0) if ball_x is not None else ((0, 255, 255) if search_mode else (0, 0, 255))
                status_text = "TRACKING" if ball_x is not None else ("SEARCHING" if search_mode else "NO DETECTION")
                
                cv2.putText(frame_bgr, f'Status: {status_text}', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                cv2.putText(frame_bgr, f'Pan: {self.pan_pos:5.1f}°  Tilt: {self.tilt_pos:5.1f}°', 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame_bgr, f'Target: {self.target_class}', 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(frame_bgr, f'Device: {self.camera_device}', 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Display frame
                cv2.imshow('Ball Tracker - Press Q to quit', frame_bgr)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('c'):
                    print("Centering servos...")
                    self.move_servos(90, 90)
                    search_mode = False
                elif key == ord('s'):
                    search_mode = not search_mode
                    print(f"Search mode: {'ON' if search_mode else 'OFF'}")
                elif key == ord('r'):
                    print("Resetting camera connection...")
                    self.camera.release()
                    time.sleep(1)
                    self.camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
                    if not self.camera.isOpened():
                        self.camera = cv2.VideoCapture(0)
                
                frame_count += 1
                time.sleep(0.03)  # ~30 FPS
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        self.tracking_active = False
        
        # Center servos before exit
        try:
            print("Centering servos...")
            self.kit.servo[0].angle = 90
            self.kit.servo[1].angle = 90
            time.sleep(1)
        except Exception as e:
            print(f"Error centering servos: {e}")
        
        # Release camera and close windows
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        print("Cleanup complete!")

def main():
    # Configuration
    MODEL_PATH = "/home/thiago/runs/detect/train3/weights/best.pt"
    TARGET_CLASS = "sports ball"  # Change this to match your YOLO model's class name
    CAMERA_DEVICE = "/dev/video0"  # Specific camera device
    
    print("=== YOLO Ball Tracker with Servo Control ===")
    print(f"Model: {MODEL_PATH}")
    print(f"Target: {TARGET_CLASS}")
    print(f"Camera: {CAMERA_DEVICE}")
    print("Hardware: Adafruit ServoKit (16-channel)")
    print("Pan Servo: Channel 0, Tilt Servo: Channel 1")
    print("=" * 45)
    
    try:
        # Create and run tracker
        tracker = BallTracker(MODEL_PATH, TARGET_CLASS, CAMERA_DEVICE)
        tracker.track_ball()
        
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please check the path to your YOLO model.")
    except Exception as e:
        print(f"Error initializing tracker: {e}")
        print("Make sure your camera and servo hardware are connected properly.")
        print("\nTroubleshooting tips:")
        print("- Check if camera is available: ls /dev/video*")
        print("- Test camera: v4l2-ctl --device=/dev/video0 --info")
        print("- Check camera permissions: sudo usermod -a -G video $USER")

if __name__ == "__main__":
    main()
