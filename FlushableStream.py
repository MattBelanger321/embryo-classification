import sys
import time
from threading import Timer

class FlushableStream:
    def __init__(self, file_name, flush_interval=5):
        self.file = open(file_name, 'w',encoding='utf-8')
        self.flush_interval = flush_interval  # Time in seconds between flushes
        self._flush_timer = None
        self._start_timer()

    def write(self, message):
        self.file.write(message)  # Write to the file
        sys.__stdout__.write(message)  # Optionally print to the console

    def flush(self):
        self.file.flush()  # Manually flush to the file
        sys.__stdout__.flush()  # Optionally flush the console output

    def _start_timer(self):
        """ Start a timer to flush periodically """
        self._flush_timer = Timer(self.flush_interval, self.flush)
        self._flush_timer.start()

    def close(self):
        """ Close the stream and stop the flush timer """
        self.flush()  # Ensure flushing before closing
        self.file.close()
        self._flush_timer.cancel()

