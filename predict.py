import cv2
from ultralytics import YOLO

class YOLOTracker:
    def __init__(self, model_path, video_source=0, test_mode=False):
        # Load the YOLO model
        self.model = YOLO(model_path)
        
        self.test_mode = test_mode
        if not self.test_mode:
            # Open the video file or stream (use 0 for default webcam)
            self.cap = cv2.VideoCapture(video_source)
            
            # Set the capture properties to use MJPEG format
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    def start_tracking(self):
        if self.test_mode:
            # Read the test image
            frame = cv2.imread("test.jpg")
            if frame is None:
                print("Error: test.jpg not found.")
                return

            # Run YOLO tracking on the frame
            results = self.model.track(frame, persist=True)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            if len(results[0].boxes) > 0:
                class_tensor = results[0].boxes.cls
                class_id = class_tensor[0].item()
                class_name = self.model.names[class_id]
                print(class_name)

            # Display the annotated frame
            cv2.imshow("YOLO Tracking", annotated_frame)
            cv2.waitKey(0)  # Wait for a key press to close the window
            cv2.destroyAllWindows()
            
        else:
            # Loop through the video frames
            while self.cap.isOpened():
                # Read a frame from the video
                success, frame = self.cap.read()

                if success:
                    # Run YOLO tracking on the frame, persisting tracks between frames
                    results = self.model.track(frame, persist=True)

                    # Visualize the results on the frame
                    annotated_frame = results[0].plot()
                    if len(results[0].boxes) > 0:
                        class_id = results[0].boxes.cls
                        class_name = self.model.names[class_id]
                        print(class_name)

                    # Display the annotated frame
                    cv2.imshow("YOLO Tracking", annotated_frame)

                    # Break the loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                else:
                    # Break the loop if the end of the video is reached
                    break

            # Release the video capture object and close the display window
            self.cap.release()
            cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    import sys
    # test_mode = "--test" in sys.argv
    test_mode = True
    tracker = YOLOTracker("runs/detect/train1/weights/best.pt", test_mode=test_mode)
    tracker.start_tracking()
