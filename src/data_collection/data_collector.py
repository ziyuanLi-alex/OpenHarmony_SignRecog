import cv2
import os
from datetime import datetime
import numpy as np
import sys
# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now you can import using the absolute path
from devices.BerxelSdkDriver.BerxelHawkContext import *


class DataCollector:
    def __init__(self, video_source=0, output_dir='./dataset/train/images'):
        self.video_source = video_source  # Can be a camera ID or video file path
        self.output_dir = output_dir
        self.__context = None
        self.__device = None
        self.__deviceList = []
        self.current_letter = None

    def openDevice(self):
        """Open device and initialize camera."""
        self.__context = BerxelHawkContext()
        if self.__context is None:
            print("Context initialization failed.")
            return False

        self.__context.initCamera()
        self.__deviceList = self.__context.getDeviceList()

        if len(self.__deviceList) < 1:
            print("Cannot find device")
            return False

        self.__device = self.__context.openDevice(self.__deviceList[0])

        if self.__device is None:
            print("Failed to open device")
            return False

        print("Device successfully opened.")
        return True



    def startStream(self):
        """Start the RGB and Depth streams."""
        if self.__device is None:
            print("Device not initialized.")
            return False

        # Start both RGB and Depth streams
        ret = self.__device.startStreams(
            BerxelHawkStreamType.forward_dict['BERXEL_HAWK_DEPTH_STREAM'] |
            BerxelHawkStreamType.forward_dict['BERXEL_HAWK_COLOR_STREAM']
        )

        if ret == 0:
            print("Started RGB and Depth streams successfully.")
            return True
        else:
            print("Failed to start streams.")
            return False

    def collect_images(self):
        """Capture and classify images based on the pressed letter."""
        while True:
            # Capture the RGB frame
            ret, frame = self.__device.readColorFrame(30)
            if not ret or frame is None:  # Check if frame is None
                print("Failed to capture RGB frame")
                continue  # Skip this iteration if no frame was captured

            # Capture the Depth frame
            depth_frame = self.__device.readDepthFrame(30)
            if depth_frame is None:
                print("Failed to capture Depth frame")
                continue  # Skip this iteration if no depth frame was captured

            # Wait for a key press to trigger capture
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC key to exit
                break

            # Check if a letter key is pressed (A-Z)
            if 65 <= key <= 90:  # ASCII values for 'A' to 'Z'
                self.current_letter = chr(key)  # Set the current letter
                print(f"Capturing image for letter: {self.current_letter}")

                # Generate timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

                # Save RGB and Depth images separately
                self.save_rgb_image(frame, timestamp)
                self.save_depth_image(depth_frame, timestamp)


    def save_rgb_image(self, frame, timestamp):
        """Save the RGB image."""
        # Create folder for the current letter if it doesn't exist
        letter_folder = os.path.join(self.output_dir, self.current_letter)
        if not os.path.exists(letter_folder):
            os.makedirs(letter_folder)

        # Save the RGB image
        image_filename = f"{self.current_letter}_rgb_{timestamp}.jpg"
        image_path = os.path.join(letter_folder, image_filename)
        cv2.imwrite(image_path, frame)

    def save_depth_image(self, depth_frame, timestamp):
        """Save the Depth image."""
        # Create folder for the current letter if it doesn't exist
        letter_folder = os.path.join(self.output_dir, self.current_letter)
        if not os.path.exists(letter_folder):
            os.makedirs(letter_folder)

        # Convert depth frame to a displayable format
        depth_array = np.ndarray(shape=(depth_frame.getHeight(), depth_frame.getWidth()), dtype=np.uint16, buffer=depth_frame.getDataAsUint16())
        depth_image = (depth_array / 10000.0 * 255).astype(np.uint8)  # Normalize depth image to [0, 255]

        # Save the Depth image
        depth_filename = f"{self.current_letter}_depth_{timestamp}.jpg"
        depth_path = os.path.join(letter_folder, depth_filename)
        cv2.imwrite(depth_path, depth_image)

    def release(self):
        """Release the camera and close the device."""
        if self.__device is not None:
            self.__device.release()
        if self.__context is not None:
            self.__context.destroyCamera()

if __name__ == '__main__':
    collector = DataCollector(video_source=0)  # Replace with video file path if needed
    if collector.openDevice():
        if collector.startStream():  # Start both RGB and Depth streams
            collector.collect_images()
    collector.release()
