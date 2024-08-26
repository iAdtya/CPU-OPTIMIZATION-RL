
import numpy as np
from .scheduler import Scheduler
from collections import deque

class MLQ(Scheduler):
    def __init__(self, data, **kwargs):
        super().__init__(data=data)
        self.queues = [deque(), deque(), deque()]  # Three priority queues

    def run(self):
        time = 0
        while any(queue for queue in self.queues) or self.data.size > 0:
            # Enqueue new arrivals to appropriate queues
            while self.data.size > 0 and self.data[0, 1] <= time:
                priority = np.random.randint(0,2)  # Assuming priority is in the 4th column
                self.queues[priority].append(self.data[0])
                self.data = np.delete(self.data, 0, axis=0)

            # Process the highest priority non-empty queue
            do_work = False
            for queue in self.queues:
                if queue:
                    do_work = True
                    process = queue.popleft()
                    self.gantt.append(int(process[0]))
                    process[2] -= 1  # Decrement remaining time

                    if process[2] > 0:
                        queue.appendleft(process)  # Requeue the process
                    break  # Process only one process at a time
            
            if do_work == False:
                self.gantt.append(-1)  # If all queues are empty, log idle time

            time += 1  # Increment the global time
        return
