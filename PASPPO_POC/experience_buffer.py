"""
Data structure for saving experience item(State, action, next state, reward...)
will sample only from the agents who have been working longer than duration. 

"""

from collections import deque
from collections import namedtuple

import random
import numpy as np

ExperienceRecord = namedtuple("ExperienceRecord", "state, action, pi, nextstate, reward, duration, doneflag, legal_acts")  

class TimedBuffer(object):
    def __init__(self, timecap):
        self.timecap = timecap
        self.buffer = deque()

    def duration(self):
        sum = 0.0
        for i in range(len(self.buffer)):
            sum += self.buffer[i].duration
        return sum

    def hasdone(self):
        for i in range(len(self.buffer)):
            if self.buffer[i].doneflag > 0:
                return True
        return False
    def add(self, exp):
        self.buffer.append(exp)
        dur = self.duration()
        while dur - self.buffer[0].duration > self.timecap:
            dur = dur - self.buffer[0].duration
            self.buffer.popleft()

class SegregatedExperienceBuffer(object):

    def __init__(self, timecap):
        self.timecap = timecap
        self.buffers = {}

    def add(self, sid, experience):
        try: 
            self.buffers[sid].add(experience)
        except KeyError:
            self.buffers[sid] = TimedBuffer(self.timecap)
            self.buffers[sid].add(experience)

    def __len__(self):
        return len(self.buffers)

    #choose a random eligible stream and return the latest segment that is at least duration long or 
    #which ends in a doneflag and is at least duration long or starts at the beginning of the buffer 
    # or immediately after previous doneflag
    def sample(self, duration):
        eligible = {}
        for sid in self.buffers:
            if self.buffers[sid].hasdone or self.buffers[sid].duration > duration:
                eligible[sid] =  self.buffers[sid]
        if len(eligible) == 0:
            return []
        tb = eligible[random.choice(list(eligible.keys()))]
        stop = len(tb.buffer)
        start = stop - 1
        dur = tb.buffer[start].duration
        while dur < duration and start > 0:
            if tb.buffer[start].doneflag > 0 and tb.buffer[stop -1].doneflag <=0:
                stop = stop -1
                dur = dur - tb.buffer[stop].duration 
            start = start - 1
            dur = dur + tb.buffer[start].duration
        return [tb.buffer[i] for i in range(start, stop)]
