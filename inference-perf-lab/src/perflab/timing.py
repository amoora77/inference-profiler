import time


class Timer:
    def __init__(self):
        self.segments = {}
        self.current_segment = None
        self.start_time = None

    def start_segment(self, name):
        if self.current_segment:
            self.end_segment()
        self.current_segment = name
        self.start_time = time.perf_counter()

    def end_segment(self):
        if self.current_segment is None:
            return
        elapsed = (time.perf_counter() - self.start_time) * 1000
        if self.current_segment not in self.segments:
            self.segments[self.current_segment] = []
        self.segments[self.current_segment].append(elapsed)
        self.current_segment = None
        self.start_time = None

    def get_totals(self):
        totals = {}
        for name, times in self.segments.items():
            totals[name] = sum(times)
        return totals

    def get_means(self):
        means = {}
        for name, times in self.segments.items():
            means[name] = sum(times) / len(times) if times else 0.0
        return means

    def get_iterations(self, segment_name):
        return self.segments.get(segment_name, [])
