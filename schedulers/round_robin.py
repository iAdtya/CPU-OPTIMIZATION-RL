from .scheduler import Scheduler
from collections import deque

class RoundRobin (Scheduler):
    def __init__ (self, data, **kwargs):
        super().__init__(data=data)
    
    def run(self):
        deq = deque()
        time = 0
        quantum = 4 # time quantum in RR scheduling
        consecutive = 0 # counter for number of times a process is run consecutively

        while len(deq) != 0 or self.data.size > 0: # if queue is not empty or data is not empty

            while self.data.size > 0 and self.data[0,1] == time: # add all processes to queue that arrive at 'time'
                deq.append(self.data[0])
                self.data = self.data[1:]

            if deq: # if queue is not empty
                process = deq.popleft()
                self.gantt.append(int(process[0])) # store PID in gantt
                process[2] -= 1 # decrease InstructionCount
                
                if process[2] == 0: # if task is complete, move on
                    consecutive = 0
                    time += 1
                    continue

                consecutive += 1
                if consecutive != quantum: # if quantum is not reached, add to front of queue
                    deq.appendleft(process)
                else: # else add to end of queue

                    # some rough code below to fix an edge case where quantum is reached and task is added at the same time
                    # we want the new task to be added before the quantum limited task
                    time += 1
                    while self.data.size > 0 and self.data[0,1] == time:
                        deq.append(self.data[0])
                        self.data = self.data[1:]
                    deq.append(process)

                    consecutive = 0
                    continue
            
            else: # if qeueue is empty, then store -1 in the gantt chart (list)
                self.gantt.append(-1)

            time += 1
        return