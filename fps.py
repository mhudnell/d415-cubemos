import datetime
from collections import deque

class FPS:
    def __init__(self, deque_size=2, update_rate=15):
        # deque_size: the number of time periods to average over
        # update_rate: the rate (number of frames) at which the caller plans to call the update method
        self._update_rate = update_rate
        self._time_deque = deque([], maxlen=deque_size)


    def update(self):
        # To be called at the interval specified by `update_rate`. Caller must ensure they call this method at the correct intervals.
        # add the current time to the time deque
        curr_time = datetime.datetime.now()
        self._time_deque.append(curr_time)

    def elapsed(self):
        # return the total number of seconds between the start and end interval
        return (self._time_deque[-1] - self._time_deque[0]).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        if len(self._time_deque) < 2:
            return -1
        else:
            return ((self._time_deque.maxlen-1) * self._update_rate) / self.elapsed()
