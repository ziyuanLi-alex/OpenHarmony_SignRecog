import psutil
import torch
import time

class PerformanceMonitor:
    def __init__(self, gpu_memory_threshold='3GB', ram_threshold='16GB', fps_target=30):
        """
        Initialize the PerformanceMonitor to track system resource usage.

        :param gpu_memory_threshold: Threshold for GPU memory usage (e.g., '3GB')
        :param ram_threshold: Threshold for RAM usage (e.g., '16GB')
        :param fps_target: Target FPS (frames per second) for the system
        """
        self.gpu_memory_threshold = self._parse_memory_limit(gpu_memory_threshold)
        self.ram_threshold = self._parse_memory_limit(ram_threshold)
        self.fps_target = fps_target

        self.last_frame_time = time.time()  # To calculate FPS

    def _parse_memory_limit(self, memory_str):
        """
        Parse the memory limit from a string (e.g., '3GB' -> 3 * 1024MB).
        
        :param memory_str: Memory limit string like '3GB'
        :return: Parsed memory in MB
        """
        if 'GB' in memory_str:
            return int(memory_str.replace('GB', '')) * 1024  # Convert GB to MB
        elif 'MB' in memory_str:
            return int(memory_str.replace('MB', ''))
        return 0  # Default to no limit

    def get_gpu_memory_usage(self):
        """
        Returns the current GPU memory usage in MB (using PyTorch).
        
        :return: GPU memory usage in MB
        """
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2  # Convert from bytes to MB
        return 0  # Return 0 if no GPU is available

    def get_ram_usage(self):
        """
        Returns the current RAM usage in MB.
        
        :return: RAM usage in MB
        """
        return psutil.virtual_memory().used / 1024**2  # Convert from bytes to MB

    def get_fps(self):
        """
        Calculate and return the current FPS.
        
        :return: FPS
        """
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_frame_time)  # FPS = 1 / time between frames
        self.last_frame_time = current_time
        return fps

    def should_skip_frame(self):
        """
        Check whether the current frame should be skipped based on the resource usage and target FPS.

        :return: True if the frame should be skipped, False otherwise.
        """
        gpu_memory_usage = self.get_gpu_memory_usage()
        ram_usage = self.get_ram_usage()
        fps = self.get_fps()

        if gpu_memory_usage > self.gpu_memory_threshold:
            print(f"Warning: GPU memory usage ({gpu_memory_usage}MB) exceeds threshold ({self.gpu_memory_threshold}MB).")
            return True

        if ram_usage > self.ram_threshold:
            print(f"Warning: RAM usage ({ram_usage}MB) exceeds threshold ({self.ram_threshold}MB).")
            return True

        if fps < self.fps_target:
            print(f"Warning: FPS ({fps:.2f}) is below the target ({self.fps_target}).")
            return True

        return False
