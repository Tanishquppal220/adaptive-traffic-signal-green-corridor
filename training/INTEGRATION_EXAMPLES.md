# Traffic Detection Integration Examples

This document shows how to integrate the trained YOLOv8 model into the main application.

## Quick Start

### Load and Run Inference
```python
from ultralytics import YOLO
from pathlib import Path

# Load the trained model
model = YOLO('models/traffic_detection_yolov8n.pt')

# Run prediction on image
image_path = 'path/to/image.jpg'
results = model.predict(image_path, conf=0.5)

# Get vehicle count
vehicle_count = len(results[0].boxes)
print(f"Vehicles detected: {vehicle_count}")
```

## Module Integration

### detection/__init__.py
```python
"""Traffic vehicle detection module using YOLOv8."""

from pathlib import Path
from ultralytics import YOLO
import numpy as np
from typing import Optional, Tuple, List

class VehicleDetector:
    """Detects vehicles in images using YOLOv8n."""
    
    def __init__(self, model_path: str = 'models/traffic_detection_yolov8n.pt'):
        """Initialize detector with trained YOLOv8 model."""
        self.model = YOLO(model_path)
        self.input_size = 416
        self.conf_threshold = 0.5
    
    def detect_vehicles(self, image: np.ndarray) -> Tuple[int, float]:
        """
        Detect vehicles and return count and average confidence.
        
        Args:
            image: Input image (numpy array or path)
            
        Returns:
            Tuple of (vehicle_count, avg_confidence)
        """
        results = self.model.predict(image, conf=self.conf_threshold, verbose=False)
        
        boxes = results[0].boxes
        vehicle_count = len(boxes)
        
        if vehicle_count > 0:
            confidences = boxes.conf.cpu().numpy()
            avg_confidence = float(np.mean(confidences))
        else:
            avg_confidence = 0.0
        
        return vehicle_count, avg_confidence
    
    def get_bounding_boxes(self, image: np.ndarray) -> List[dict]:
        """
        Get detailed bounding box information for detected vehicles.
        
        Returns:
            List of dicts with keys: 'x1', 'y1', 'x2', 'y2', 'confidence'
        """
        results = self.model.predict(image, conf=self.conf_threshold, verbose=False)
        
        boxes = results[0].boxes
        detections = []
        
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf)
            
            detections.append({
                'x1': int(xyxy[0]),
                'y1': int(xyxy[1]),
                'x2': int(xyxy[2]),
                'y2': int(xyxy[3]),
                'confidence': confidence
            })
        
        return detections
    
    def process_video_frame(self, frame: np.ndarray) -> dict:
        """
        Process a single video frame and return statistics.
        
        Returns:
            Dict with keys: 'count', 'avg_confidence', 'detections'
        """
        count, avg_conf = self.detect_vehicles(frame)
        detections = self.get_bounding_boxes(frame)
        
        return {
            'count': count,
            'avg_confidence': avg_conf,
            'detections': detections
        }


# Export main class
__all__ = ['VehicleDetector']
```

### control/__init__.py Integration
```python
"""Traffic signal control with vehicle detection."""

from detection import VehicleDetector
import numpy as np
from typing import Dict

class TrafficSignalController:
    """Adaptive traffic signal control using detected vehicle density."""
    
    def __init__(self):
        self.detector = VehicleDetector()
        self.max_green_time = 120  # seconds
        self.min_green_time = 15   # seconds
    
    def get_signal_timing(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Calculate optimal signal timing based on detected vehicles.
        
        Args:
            frame: Current camera frame
            
        Returns:
            Dict with timing for each direction
        """
        # Detect vehicles
        detection_result = self.detector.process_video_frame(frame)
        vehicle_count = detection_result['count']
        
        # Adaptive logic
        if vehicle_count == 0:
            green_time = self.min_green_time
        elif vehicle_count > 50:
            green_time = self.max_green_time
        else:
            # Linear interpolation
            green_time = self.min_green_time + (
                (vehicle_count / 50) * (self.max_green_time - self.min_green_time)
            )
        
        return {
            'direction_ns': green_time,
            'direction_ew': self.max_green_time - green_time,
        }
    
    def process_camera_feed(self, frames: list) -> list:
        """
        Process multiple frames and return statistics.
        
        Args:
            frames: List of numpy arrays (video frames)
            
        Returns:
            List of detection results
        """
        results = []
        for frame in frames:
            result = self.detector.process_video_frame(frame)
            results.append(result)
        
        return results


# Export main class
__all__ = ['TrafficSignalController']
```

## Video Processing Example

```python
import cv2
from detection import VehicleDetector

def process_video(video_path: str, output_path: str):
    """Process video and save results with drawn bounding boxes."""
    
    detector = VehicleDetector()
    cap = cv2.VideoCapture(video_path)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect vehicles
        results = detector.model.predict(frame, conf=0.5, verbose=False)
        
        # Draw bounding boxes
        annotated_frame = results[0].plot()
        
        # Get vehicle count
        vehicle_count = len(results[0].boxes)
        
        # Add text annotation
        cv2.putText(
            annotated_frame,
            f"Vehicles: {vehicle_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        out.write(annotated_frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Processed frame {frame_count}, vehicles: {vehicle_count}")
    
    cap.release()
    out.release()
    print(f"Output saved to {output_path}")


# Usage
# process_video('input.mp4', 'output.mp4')
```

## Batch Processing

```python
from pathlib import Path
from detection import VehicleDetector
import json

def batch_analyze_images(image_dir: str, output_json: str):
    """Analyze all images in directory and save results."""
    
    detector = VehicleDetector()
    image_dir = Path(image_dir)
    results = {}
    
    for image_path in image_dir.glob('*.jpg'):
        count, confidence = detector.detect_vehicles(str(image_path))
        results[image_path.name] = {
            'vehicle_count': count,
            'avg_confidence': float(confidence)
        }
        print(f"{image_path.name}: {count} vehicles")
    
    # Save results
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_json}")
    return results


# Usage
# results = batch_analyze_images('data/test_images/', 'detection_results.json')
```

## Real-time Inference from Webcam

```python
import cv2
from detection import VehicleDetector

def webcam_detection():
    """Real-time vehicle detection from webcam."""
    
    detector = VehicleDetector()
    cap = cv2.VideoCapture(0)  # 0 = default webcam
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect vehicles
            results = detector.model.predict(frame, conf=0.5, verbose=False)
            annotated_frame = results[0].plot()
            
            # Get statistics
            vehicle_count = len(results[0].boxes)
            
            # Display
            cv2.putText(
                annotated_frame,
                f"Vehicles: {vehicle_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            cv2.imshow('Vehicle Detection', annotated_frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()


# Usage
# webcam_detection()
```

## Performance Monitoring

```python
import time
from detection import VehicleDetector
import numpy as np

class DetectionBenchmark:
    """Benchmark detection performance."""
    
    def __init__(self, model_path: str = 'models/traffic_detection_yolov8n.pt'):
        self.detector = VehicleDetector(model_path)
        self.times = []
    
    def benchmark_inference(self, image_path: str, num_runs: int = 10):
        """Measure inference speed."""
        for _ in range(num_runs):
            start = time.time()
            self.detector.detect_vehicles(image_path)
            elapsed = time.time() - start
            self.times.append(elapsed * 1000)  # Convert to ms
        
        times = np.array(self.times)
        print(f"\nBenchmark Results ({num_runs} runs):")
        print(f"  Average: {np.mean(times):.2f} ms")
        print(f"  Min:     {np.min(times):.2f} ms")
        print(f"  Max:     {np.max(times):.2f} ms")
        print(f"  Std:     {np.std(times):.2f} ms")
        print(f"  FPS:     {1000/np.mean(times):.1f} fps")


# Usage
# benchmark = DetectionBenchmark()
# benchmark.benchmark_inference('test_image.jpg')
```

## Configuration & Tuning

```python
from detection import VehicleDetector

class ConfigurableDetector(VehicleDetector):
    """Detector with adjustable parameters."""
    
    def __init__(self, model_path: str = 'models/traffic_detection_yolov8n.pt',
                 conf_threshold: float = 0.5,
                 use_gpu: bool = True):
        super().__init__(model_path)
        self.conf_threshold = conf_threshold
        
        # Force device
        if use_gpu:
            self.model.to('cuda')
        else:
            self.model.to('cpu')
    
    def set_confidence_threshold(self, threshold: float):
        """Update confidence threshold for NMS."""
        self.conf_threshold = max(0, min(1, threshold))
    
    def get_model_info(self) -> dict:
        """Get model metadata."""
        return {
            'model_name': self.model.model_name,
            'model_size': self.model.model_size,
            'parameters': sum(p.numel() for p in self.model.model.parameters()),
            'input_size': self.input_size,
            'conf_threshold': self.conf_threshold
        }
```

---

## Testing

```python
import unittest
from detection import VehicleDetector
import numpy as np

class TestVehicleDetector(unittest.TestCase):
    """Unit tests for VehicleDetector."""
    
    @classmethod
    def setUpClass(cls):
        cls.detector = VehicleDetector()
    
    def test_empty_image(self):
        """Test detection on empty image."""
        empty_image = np.zeros((416, 416, 3), dtype=np.uint8)
        count, conf = self.detector.detect_vehicles(empty_image)
        self.assertEqual(count, 0)
        self.assertEqual(conf, 0.0)
    
    def test_output_types(self):
        """Test output types are correct."""
        count, conf = self.detector.detect_vehicles('sample.jpg')
        self.assertIsInstance(count, int)
        self.assertIsInstance(conf, float)
    
    def test_confidence_range(self):
        """Test confidence is in valid range."""
        count, conf = self.detector.detect_vehicles('sample.jpg')
        self.assertTrue(0 <= conf <= 1)


if __name__ == '__main__':
    unittest.main()
```

---

## See Also

- [Training Guide](TRAINING_GUIDE.md)
- [Implementation Summary](IMPLEMENTATION_SUMMARY.md)
- [Project Architecture](../../.github/copilot-instructions.md)
- [YOLOv8 Documentation](https://docs.ultralytics.com)
