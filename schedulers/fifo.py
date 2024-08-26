from .scheduler import Scheduler
from collections import deque

class FIFO(Scheduler):
    def __init__(self, data, **kwargs):
        super().__init__(data=data)

    def run(self):
        deq = deque()
        time = 0

        while len(deq) != 0 or self.data.size > 0: # if queue is not empty or data is not empty

            while self.data.size > 0 and self.data[0,1] == time: # add all processes to queue that arrive at 'time'
                deq.append(self.data[0])
                self.data = self.data[1:]

            if deq: # if queue is not empty
                process = deq.popleft()
                self.gantt.append(int(process[0])) # store PID in gantt
                process[2] -= 1 # decrease InstructionCount
                
                if process[2] == 0: # if task is complete, move on
                    time += 1
                    continue
                else: # else add it to the front of queue
                    deq.appendleft(process) 

    
            else: # if qeueue is empty, then store -1 in the gantt chart (list)
                self.gantt.append(-1)

            time += 1
        return
