# Driver Monitoring System

This project implements a driver monitoring system using computer vision to detect and analyze driver behavior. The system is containerized using Docker and consists of three main components: capture, detection, and results services.

## Project Structure

```
driver-monitoring/
├── capture/         # Video capture service
├── detection/       # Driver behavior detection service
├── results/         # Results processing and storage service
├── shared/          # Shared resources and data
└── docker-compose.yml
```

## Prerequisites

- Docker and Docker Compose
- Linux-based system (for video device access)
- Webcam or camera device

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/driver-monitoring.git
   cd driver-monitoring
   ```

2. Ensure your camera device is properly connected and accessible at `/dev/video0`

3. Build and start the services:
   ```bash
   docker-compose up --build
   ```

## Services

### Capture Service
- Port: 5555
- Handles video capture from the camera device
- Requires access to `/dev/video0`

### Detection Service
- Port: 5556
- Processes video frames for driver behavior detection
- Depends on the capture service

### Results Service
- Processes and stores detection results
- Mounts the shared results directory

## Usage

1. The system will automatically start all services when you run `docker-compose up`
2. The capture service will begin streaming video from the camera
3. The detection service will process the video feed for driver monitoring
4. Results will be stored in the `shared/results` directory

## Stopping the System

To stop all services:
```bash
docker-compose down
```

## Notes

- The system requires privileged access to the video device
- All services are configured to restart automatically unless explicitly stopped
- Results are persisted in the shared directory

## License

[Add your license information here] 