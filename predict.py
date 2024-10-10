import cv2
from ultralytics import YOLO
import sys
import requests

class YOLOTracker:
    def __init__(self, model_path, video_source=0, test_mode=False, test_post=False):
        # Load the YOLO model
        self.model = YOLO(model_path)
        self.test_mode = test_mode
        self.test_post = test_post
        self.video_source = video_source
        self.latest_frame = None

        if not self.test_mode:
            # Open the video file or stream (use 0 for default webcam)
            self.cap = cv2.VideoCapture(video_source)
            if not self.cap.isOpened():
                raise ValueError(f"Error: Unable to open video source {video_source}")
            # Set the capture properties to use MJPEG format
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        
        # Parameters for filtering
        self.previous_class_name = None
        self.stable_frame_count = 0
        self.required_stable_frames = 3  # Number of frames the detection needs to be stable

    def process_frame(self, frame):
        # Run YOLO tracking on the frame
        results = self.model.track(frame, persist=True, verbose=False)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        detected_class_name = None

        # If any object is detected, extract the class name
        if len(results[0].boxes) > 0:
            class_tensor = results[0].boxes.cls
            class_id = class_tensor[0].item()
            detected_class_name = self.model.names[class_id]

        # Filtering logic to prevent rapid jumps
        if detected_class_name == self.previous_class_name:
            self.stable_frame_count += 1
        else:
            self.stable_frame_count = 0

        if self.stable_frame_count >= self.required_stable_frames:
            filtered_class_name = detected_class_name
        else:
            filtered_class_name = self.previous_class_name

        # Update previous class name
        self.previous_class_name = detected_class_name

        if filtered_class_name is not None:
            print(f"Detected: {filtered_class_name}", end='\r')

        self.latest_frame = annotated_frame
        return annotated_frame, filtered_class_name

    def start_tracking(self):
        try:
            if self.test_mode:
                # Read the test image
                frame = cv2.imread("test2.jpg")
                if frame is None:
                    print("Error: test.jpg not found.", end='\r')
                    return

                # Process the frame
                annotated_frame, class_name = self.process_frame(frame)

                if class_name:
                    self.post_class_name(class_name)

                # Display the annotated frame
                cv2.imshow("YOLO Tracking", annotated_frame)
                cv2.waitKey(0)  # Wait for a key press to close the window
            else:
                # Loop through the video frames
                while self.cap.isOpened():
                    # Read a frame from the video
                    success, frame = self.cap.read()

                    if not success:
                        # Break the loop if the end of the video is reached
                        break

                    # Process the frame
                    annotated_frame, class_name = self.process_frame(frame)

                    # Post the class_name using POST if detected
                    if class_name:
                        self.post_class_name(class_name)

                    # Display the annotated frame
                    cv2.imshow("YOLO Tracking", annotated_frame)

                    # Break the loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            # Release the video capture object and close the display window
            if not self.test_mode:
                self.cap.release()
            cv2.destroyAllWindows()

    def post_class_name(self, class_name):
        url = "http://localhost:5000/recognize" 
        data = {"class_name": class_name}
        
        if self.test_post:
            print(f"Pseudo-posting data: {data}", end='\r')
        else:
            try:
                response = requests.post(url, json=data)
                if response.status_code == 200:
                    # print("Posted successfully", end='\r')
                    pass
                else:
                    print(f"Failed to post: {response.status_code}", end='\r')
            except requests.exceptions.RequestException as e:
                print(f"Error posting data: {e}", end='\r')
                
    def broadcast_class_name(self, class_name):
        socketio.emit('recognized_class', {'class_name': class_name}, broadcast=True)

    def get_latest_frame(self):
        return self.latest_frame

# Example usage
if __name__ == "__main__":
    # test_mode = "--test" in sys.argv
    tracker = YOLOTracker("runs/detect/train1/weights/best.pt", test_mode=False)
    tracker.start_tracking()