from typing import Iterable


def chunks(items: Iterable, size: int):
    """Yield successive size-sized chunks from an indexable iterable."""
    for index in range(0, len(items), size):
        yield items[index:index + size]


class AvgMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def __call__(self, value=None):
        if value is not None:
            self.sum += value
            self.count += 1
        if self.count > 0:
            return self.sum / self.count
        return 0
