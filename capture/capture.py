import cv2
import time
import zmq

def main():
    # ZeroMQ setup
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.bind("tcp://*:5555")
    
    # Webcam setup
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Cannot open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("📸 Capture service started. Publishing frames via ZeroMQ...")
    
    try:
        frame_count = 0
        while True:
            start_time = time.time()
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Frame capture failed")
                time.sleep(0.1)
                continue
            
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            jpeg_data = buffer.tobytes()
            
            # Prepare and send multipart message
            timestamp = str(time.time()).encode('utf-8')
            publisher.send_multipart([timestamp, jpeg_data])
            frame_count += 1
            
            if frame_count % 10 == 0:
                print(f"📤 Sent frame {frame_count}")
                
            # Maintain ~5 FPS
            elapsed = time.time() - start_time
            time.sleep(max(0, 0.2 - elapsed))
            
    except KeyboardInterrupt:
        print("\n🛑 Capture stopped by user")
    except Exception as e:
        print(f"🔥 Capture error: {e}")
    finally:
        cap.release()
        publisher.close()
        context.term()
        print("✅ Resources released")

if __name__ == "__main__":
    main()

