from .scheduler import Scheduler
from collections import deque
import numpy as np
import math
import random

class CFS(Scheduler):
    def __init__(self, data, **kwargs):
        super().__init__(data=data)

    def run(self):
        zeros_col = np.zeros((self.data.shape[0], 1)) # add a column for vruntime
        self.data = np.hstack((self.data, zeros_col))
        #print(self.data)

        deq = deque()
        time = 0
        quantum = 30
        consecutive = 0 # counter for number of times a process is run consecutively
        
        while len(deq) != 0 or self.data.size > 0: # if queue is not empty or data is not empty

            while self.data.size > 0 and self.data[0,1] == time: # add all processes to queue that arrive at 'time'
                row = self.data[0]
                row[3] = random.randint(-19,19) # niceness factor added onto a initial vruntime of 0
                deq.append(row)
                self.data = self.data[1:]

            if deq:
                queue_size = len(deq)
                time_slice = math.ceil(quantum/queue_size) # dynamic timeslice

                np_arr = np.array(deq)
                sorted_np_arr = np_arr[np_arr[:,3].argsort()] # sort the queue based on vruntime
                deq = deque(sorted_np_arr)

                process = deq.popleft()
                self.gantt.append(int(process[0])) # store PID in gantt
                process[2] -= 1 # decrement instructionCount
                process[3] += 1 # incremement vruntime of active process

                if process[2] == 0: # if task is complete, move on
                    consecutive = 0
                    time += 1
                    continue

                consecutive += 1
                if consecutive != time_slice: # if time_slice is not reached, add to front of queue
                    deq.appendleft(process)
                else: # else add to end of queue
                    # some rough code below to fix an edge case where time_slice is reached and task is added at the same time
                    # we want the new task to be added before the time_slice limited task
                    time += 1
                    while self.data.size > 0 and self.data[0,1] == time:
                        row = self.data[0]
                        row[3] = random.randint(-19,19) # niceness factor added onto a initial vruntime of 0
                        deq.append(row)
                        self.data = self.data[1:]

                    deq.append(process)
                    consecutive = 0
                    continue

            else: # if qeueue is empty, then store -1 in the gantt chart (list)
                self.gantt.append(-1)

            time += 1
        return
