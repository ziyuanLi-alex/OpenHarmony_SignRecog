import cv2
from ultralytics import YOLO

class YOLOTracker:
    def __init__(self, model_path, video_source=0):
        # Load the YOLO model
        self.model = YOLO(model_path)
        
        # Open the video file or stream (use 0 for default webcam)
        self.cap = cv2.VideoCapture(video_source)
        
        # Set the capture properties to use MJPEG format
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    def start_tracking(self):
        # Loop through the video frames
        while self.cap.isOpened():
            # Read a frame from the video
            success, frame = self.cap.read()

            if success:
                # Run YOLO tracking on the frame, persisting tracks between frames
                results = self.model.track(frame, persist=True)

                # Visualize the results on the frame
                annotated_frame = results[0].plot()
                print(results[0].probs)

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
    tracker = YOLOTracker("runs/detect/train1/weights/best.pt")
    tracker.start_tracking()
