from .scheduler import Scheduler
from collections import deque
import numpy as np

class MFQ(Scheduler):
    def __init__(self, data, **kwargs):
        super().__init__(data=data)
        self.queues = [deque() for _ in range(3)]  # Three levels of queues
        self.quantums = [4, 8, 15]  # Different time quantum for each queue level

    def run(self):
        time = 0
        waiting_time_threshold = 10  # Threshold to boost the priority of a process
        while any(self.queues) or len(self.data) > 0:
            # Enqueue new arrivals to the highest priority queue
            while len(self.data) > 0 and self.data[0][1] <= time:
                self.queues[0].append(list(self.data[0]))  # Add as list for mutability
                self.data = np.delete(self.data, 0, axis=0)

            # Process queues starting from highest priority
            for i in range(len(self.queues)):
                if self.queues[i]:
                    process = self.queues[i].popleft()
                    quantum = self.quantums[i]
                    execution_time = min(process[2], quantum)
                    process[2] -= execution_time  # Reduce remaining burst time
                    time += execution_time  # Increment time by the execution time used
                    self.gantt.append(process[0])  # Log process execution

                    # Check if the process is finished
                    if process[2] > 0:
                        if i < len(self.queues) - 1:
                            self.queues[i + 1].append(process)  # Move to lower priority queue
                        else:
                            self.queues[i].append(process)  # Re-enqueue in the same queue if it's the lowest
                    break  # Only one process from one queue at a time
                elif i == len(self.queues) - 1:  # If all queues were empty
                    self.gantt.append(-1)  # CPU idle
            '''
            # Priority boost for waiting processes
            for i in range(1, len(self.queues)):
                to_promote = []
                for process in self.queues[i]:
                    if process[4] >= waiting_time_threshold:  # Assuming process[4] tracks waiting time
                        to_promote.append(process)
                for p in to_promote:
                    self.queues[i].remove(p)
                    self.queues[i - 1].append(p)  # Move to higher priority queue
            '''

        return