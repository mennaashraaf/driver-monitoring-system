import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
import zmq
import time
import json

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh

# Landmark indices
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_IRIS_INDICES = [474, 475, 476, 477]
RIGHT_IRIS_INDICES = [469, 470, 471, 472]
MOUTH_INDICES = [78, 81, 13, 311, 308, 402]

HEAD_POINTS = [
    [0.0, 0.0, 0.0],
    [0.0, -330.0, -65.0],
    [-225.0, 170.0, -135.0],
    [225.0, 170.0, -135.0],
    [-150.0, -150.0, -125.0],
    [150.0, -150.0, -125.0]
]

# Thresholds
EAR_THRESHOLD = 0.21
GAZE_THRESHOLD = 0.4
YAWN_THRESHOLD = 0.5
HEAD_YAW_THRESH = 25
HEAD_PITCH_THRESH = 25

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(landmarks_px):
    """Calculate mouth aspect ratio using pixel coordinates"""
    try:
        # Use specific mouth landmark indices
        top_lip = landmarks_px[13]      # Upper lip center
        bottom_lip = landmarks_px[14]   # Lower lip center  
        left_corner = landmarks_px[78]  # Left mouth corner
        right_corner = landmarks_px[308] # Right mouth corner
        
        vertical = dist.euclidean(top_lip, bottom_lip)
        horizontal = dist.euclidean(left_corner, right_corner)
        
        if horizontal == 0:
            return 0
        return vertical / horizontal
    except Exception as e:
        print(f"⚠️ Mouth aspect ratio error: {e}")
        return 0

def get_head_pose(image, landmarks, img_w, img_h):
    try:
        size = image.shape
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        
        # Convert normalized coordinates to pixel coordinates
        image_points = np.array([
            [landmarks[1].x * img_w, landmarks[1].y * img_h],   # Nose tip (landmark 1)
            [landmarks[152].x * img_w, landmarks[152].y * img_h], # Chin
            [landmarks[263].x * img_w, landmarks[263].y * img_h], # Left eye
            [landmarks[33].x * img_w, landmarks[33].y * img_h],  # Right eye
            [landmarks[287].x * img_w, landmarks[287].y * img_h], # Left mouth
            [landmarks[57].x * img_w, landmarks[57].y * img_h]  # Right mouth
        ], dtype="double")

        model_points = np.array(HEAD_POINTS, dtype="double")
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, None)

        if success:
            rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
            proj_matrix = np.hstack((rvec_matrix, translation_vector))
            euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
            pitch, yaw, roll = euler_angles[0], euler_angles[1], euler_angles[2]
            return float(pitch), float(yaw), float(roll)
        else:
            return 0.0, 0.0, 0.0
        
    except Exception as e:
        print(f"⚠️ Head pose error: {e}")
        return 0.0, 0.0, 0.0

def main():
    # ZeroMQ setup
    context = zmq.Context()
    
    # Receive frames from capture
    frame_receiver = context.socket(zmq.SUB)
    frame_receiver.connect("tcp://capture:5555")
    frame_receiver.setsockopt_string(zmq.SUBSCRIBE, '')
    
    # Send results to results service
    result_sender = context.socket(zmq.PUB)
    result_sender.bind("tcp://*:5556")
    
    # Initialize face mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    print("👀 Detection service started. Waiting for frames...")
    last_state = None
    
    try:
        while True:
            # Receive frame
            try:
                # First receive as raw bytes
                raw_message = frame_receiver.recv(zmq.NOBLOCK)
                
                # Check if it's a JPEG image (starts with JPEG magic bytes)
                if raw_message.startswith(b'\xff\xd8\xff'):
                    # This is raw JPEG data
                    jpeg_data = raw_message
                    timestamp = time.time()
                    print(f"📸 Received raw JPEG frame ({len(jpeg_data)} bytes)")
                else:
                    # Try to decode as JSON
                    try:
                        message = json.loads(raw_message.decode('utf-8'))
                        if isinstance(message, dict):
                            timestamp = message.get("timestamp", time.time())
                            jpeg_data = message.get("frame")
                            
                            if jpeg_data is None:
                                print("⚠️ No frame data in JSON message")
                                continue
                                
                            # Handle different frame data formats
                            if isinstance(jpeg_data, str):
                                # If it's a base64 string, decode it
                                import base64
                                jpeg_data = base64.b64decode(jpeg_data)
                            elif isinstance(jpeg_data, list):
                                # If it's a list of bytes, convert to bytes
                                jpeg_data = bytes(jpeg_data)
                        else:
                            # Received a float or other type - might be just timestamp
                            print(f"📊 Received timestamp: {message}")
                            continue
                            
                    except json.JSONDecodeError:
                        print(f"⚠️ Unknown message format, skipping...")
                        continue
                    except UnicodeDecodeError:
                        print(f"⚠️ Binary data that's not JPEG, skipping...")
                        continue
                    
            except zmq.Again:
                time.sleep(0.01)  # Small delay to prevent busy waiting
                continue
            except Exception as e:
                print(f"⚠️ Error receiving message: {e}")
                time.sleep(0.1)
                continue
            
            # Convert to OpenCV image
            try:
                if not isinstance(jpeg_data, (bytes, bytearray)):
                    print(f"⚠️ Invalid frame data type: {type(jpeg_data)}")
                    continue
                    
                nparr = np.frombuffer(jpeg_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image is None:
                    print("⚠️ Failed to decode JPEG image")
                    continue
            except Exception as e:
                print(f"⚠️ Image decoding error: {e}")
                continue
                
            img_h, img_w, _ = image.shape
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process image
            drowsy = False
            distracted = False
            yawn = False
            
            results = face_mesh.process(image_rgb)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Get raw landmarks (normalized coordinates)
                    raw_landmarks = face_landmarks.landmark
                    
                    # Convert to pixel coordinates for eye/mouth detection
                    landmarks_px = [(int(lm.x * img_w), int(lm.y * img_h)) 
                                   for lm in raw_landmarks]

                    # Eye detection
                    try:
                        left_eye = [landmarks_px[i] for i in LEFT_EYE_INDICES]
                        right_eye = [landmarks_px[i] for i in RIGHT_EYE_INDICES]
                        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
                        if ear < EAR_THRESHOLD:
                            drowsy = True
                    except Exception as e:
                        print(f"⚠️ Eye detection error: {e}")

                    # Yawn detection
                    try:
                        mar = mouth_aspect_ratio(landmarks_px)
                        if mar > YAWN_THRESHOLD:
                            yawn = True
                            drowsy = True
                    except Exception as e:
                        print(f"⚠️ Yawn detection error: {e}")

                    # Head pose - using raw landmarks (normalized coordinates)
                    try:
                        pitch, yaw_angle, roll = get_head_pose(image, raw_landmarks, img_w, img_h)
                        if abs(yaw_angle) > HEAD_YAW_THRESH or abs(pitch) > HEAD_PITCH_THRESH:
                            distracted = True
                    except Exception as e:
                        print(f"⚠️ Head pose detection error: {e}")

                    # Gaze detection - using pixel coordinates
                    try:
                        # Check if we have enough landmarks for gaze detection
                        if len(landmarks_px) > max(max(LEFT_IRIS_INDICES), max(RIGHT_IRIS_INDICES)):
                            # Get face center using more reliable points
                            face_center_x = (landmarks_px[234][0] + landmarks_px[454][0]) / 2
                            
                            # Get iris centers
                            left_iris = np.array([landmarks_px[i] for i in LEFT_IRIS_INDICES]).mean(axis=0)
                            right_iris = np.array([landmarks_px[i] for i in RIGHT_IRIS_INDICES]).mean(axis=0)
                            
                            # Calculate gaze center
                            gaze_center = (left_iris[0] + right_iris[0]) / 2
                            
                            # Calculate horizontal offset
                            gaze_offset = (gaze_center - face_center_x) / img_w
                            
                            if abs(gaze_offset) > GAZE_THRESHOLD:
                                distracted = True
                        else:
                            print("⚠️ Not enough landmarks for gaze detection")
                    except Exception as e:
                        print(f"⚠️ Gaze detection error: {e}")

            # Create result
            current_state = (drowsy, distracted, yawn)
            result = {
                "timestamp": timestamp,
                "drowsy": drowsy,
                "distracted": distracted,
                "yawning": yawn
            }
            
            # Send result if state changed or every few iterations
            if current_state != last_state:
                try:
                    result_sender.send_json(result, zmq.NOBLOCK)
                    last_state = current_state
                    print(f"🔔 State changed: drowsy={drowsy}, distracted={distracted}, yawn={yawn}")
                except zmq.Again:
                    print("⚠️ Could not send result, receiver busy")
                    
    except KeyboardInterrupt:
        print("\n🛑 Detection stopped by user")
    except Exception as e:
        print(f"🔥 Detection error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        frame_receiver.close()
        result_sender.close()
        context.term()
        print("✅ ZeroMQ connections closed")

if __name__ == "__main__":
    main()
