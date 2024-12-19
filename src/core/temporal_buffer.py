import torch
import numpy as np
from collections import deque

class TemporalBuffer:
    def __init__(self, buffer_size=15, max_memory_usage='5GB'):
        """
        Initialize the Temporal Buffer to store frames and control memory usage.

        :param buffer_size: The number of frames to store in the buffer
        :param max_memory_usage: Maximum memory usage for the buffer (as string, e.g., '2GB')
        """

        self.buffer_size = buffer_size
        self.max_memory_usage = self._parse_memory_limit(max_memory_usage)
        self.buffer = deque(maxlen=buffer_size)  # FIFO buffer for frames
        self.current_memory_usage = 0  # Track memory usage
        self.frame_size = None  # Frame size will be determined dynamically

    def _parse_memory_limit(self, memory_str):
        """
        Parse the memory limit from a string (e.g., '2GB' -> 2 * 1024MB)
        
        :param memory_str: Memory limit string like '2GB'
        :return: Parsed memory in MB
        """
        if 'GB' in memory_str:
            return int(memory_str.replace('GB', '')) * 1024  # Convert GB to MB
        if 'MB' in memory_str:
            return int(memory_str.replace('MB', ''))
        return 0  # Default to no limit

    def _estimate_frame_memory(self, frame):
        """
        Estimate the memory usage of a single frame.
        
        :param frame: The frame to estimate memory for
        :return: Estimated memory usage in MB
        """
        frame_tensor = torch.tensor(frame)
        return frame_tensor.element_size() * frame_tensor.numel() / 1024**2  # In MB

    def update(self, rgb_features, depth_features):
        """
        Update the buffer with new features (RGB and Depth).
        
        :param rgb_features: RGB features from the YOLO model
        :param depth_features: Depth features from the YOLO model
        """
        # Estimate the memory of the incoming frame features
        rgb_memory = self._estimate_frame_memory(rgb_features)
        depth_memory = self._estimate_frame_memory(depth_features)
        
        # Check if adding the new frame will exceed memory limit
        total_memory = self.current_memory_usage + rgb_memory + depth_memory
        if total_memory > self.max_memory_usage:
            self._drop_old_frames(rgb_memory + depth_memory)

        # Add frames to the buffer
        self.buffer.append((rgb_features, depth_features))
        self.current_memory_usage += rgb_memory + depth_memory

        # Track frame size for future memory estimations
        if self.frame_size is None:
            self.frame_size = rgb_features.shape  # Assuming rgb_features and depth_features have the same shape

    def _drop_old_frames(self, memory_to_free):
        """
        Drop old frames from the buffer to free memory.
        
        :param memory_to_free: The amount of memory to free in MB
        """
        while self.current_memory_usage + memory_to_free > self.max_memory_usage:
            old_rgb, old_depth = self.buffer.popleft()
            self.current_memory_usage -= self._estimate_frame_memory(old_rgb)
            self.current_memory_usage -= self._estimate_frame_memory(old_depth)

    def get_averaged_features(self):
        """
        Return the averaged features of all frames in the buffer.
        
        :return: Averaged RGB and Depth features
        """
        if len(self.buffer) == 0:
            return None, None
        
        rgb_frames = torch.stack([torch.tensor(rgb) for rgb, _ in self.buffer])
        depth_frames = torch.stack([torch.tensor(depth) for _, depth in self.buffer])
        
        # Return the average features across the buffer
        averaged_rgb = rgb_frames.mean(dim=0)
        averaged_depth = depth_frames.mean(dim=0)
        
        return averaged_rgb, averaged_depth
