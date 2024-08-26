from .scheduler import Scheduler
from queue import PriorityQueue
from Custom_ENV.neural_network import FeedForwardNN
import numpy as np
import torch


class MLPriority(Scheduler):
    def __init__(self, data, **kwargs):
        super().__init__(data=None)
        self.data = np.zeros(shape=(data.shape[0], 4))
        self.data[:, :3] = data
        self.data[:, 3] = self.data[:, 2]
        self.pids = data[:, 0].astype(int)
        self.arrivals = data[:, 1]
        self.instr_count = data[:, 2]
        try:
            self.encoder_context = kwargs.get("encoder_context", 30)
            self.max_priority = kwargs.get("max_priority", 10)
        except NotImplementedError:
            print(
                "MLPriority must be instantiated with 'encoder_context' and 'max_priority' keyword args"
            )
        self.model = FeedForwardNN((self.encoder_context + 1) * 5, self.max_priority)
        self.model.load_state_dict(
            torch.load("model_weights/ml_priority_scheduler_5mil_30context.pt")
        )

    def run(self):
        self.execution_queue = PriorityQueue()
        time = 0

        while (
            len(self.execution_queue.queue) > 0 or self.data.size > 0
        ):  # if priority queue is not empty or data is not empty

            while (
                self.data.size > 0 and self.data[0, 1] == time
            ):  # add all processes to queue that arrrive at time
                priority = self.get_priority()
                # print(priority)
                self.execution_queue.put((priority, list(self.data[0])))
                self.data = self.data[1:]

            if len(self.execution_queue.queue) > 0:  # if queue is not empty
                priority, process = self.execution_queue.queue[0]
                self.gantt.append(int(process[0]))
                process[3] -= 1
                self.execution_queue.queue[0] = (priority, process)

                if process[3] == 0:  # task is complete
                    _ = self.execution_queue.get()
                    time += 1
                    continue

            else:  # if queue is empty, store -1 in gantt chart
                self.gantt.append(-1)

            time += 1
        return

    def get_priority(self):
        obs = self.get_observation().ravel()
        priority = np.argmax(self.model(obs).detach().cpu().numpy())
        return priority

    def get_observation(self):
        obs = np.ones((self.encoder_context + 1, 5), dtype=np.int32) * (-1)
        obs[0, :4] = self.data[0]
        for i in range(self.encoder_context):
            if i < len(self.execution_queue.queue):
                obs[i + 1, :4] = self.execution_queue.queue[i][1]
                obs[i + 1, 4] = self.execution_queue.queue[i][0]
            else:
                break
        return obs
