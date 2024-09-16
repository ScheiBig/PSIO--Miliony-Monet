import time

class stopwatch:
    def __init__(self, start=True) -> None:
        self.start_time: float | None = None
        if start:
            self.start()

    def start(self) -> None:
        self.start_time = time.time()

    def stop(self) -> float:
        if self.start_time is None:
            raise ValueError("Stopwatch has not been started.")
        end_time: float = time.time()
        elapsed_time: float = (end_time - self.start_time) * 1000  # Convert to milliseconds
        self.start_time = None  # Reset the stopwatch
        return elapsed_time
