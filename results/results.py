import zmq
import json
import os
import time

def main():
    # ZeroMQ setup
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.connect("tcp://detection:5556")
    subscriber.setsockopt_string(zmq.SUBSCRIBE, '')
    
    # Output setup
    output_dir = "/shared/results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "results.json")
    
    print("📊 Results service started. Waiting for detection results...")
    
    try:
        while True:
            # Receive result
            result = subscriber.recv_json()
            
            # Write to file
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=4)
                
            print(f"📝 Updated results: drowsy={result['drowsy']}, "
                  f"distracted={result['distracted']}, yawn={result['yawning']}")
            
    except KeyboardInterrupt:
        print("\n🛑 Results service stopped")
    except Exception as e:
        print(f"🔥 Results error: {e}")
    finally:
        subscriber.close()
        context.term()
        print("✅ ZeroMQ connection closed")

if __name__ == "__main__":
    main()
