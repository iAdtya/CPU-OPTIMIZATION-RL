import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from queue import PriorityQueue
import numpy as np


register(
    id="PrioritySchedulerEnv",
    entry_point="priority_scheduler:PrioritySchedulerEnv",
)


class PrioritySchedulerEnv(gym.Env):
    def __init__(self, data, encoder_context, max_priority) -> None:
        super(PrioritySchedulerEnv, self).__init__()

        self.data = data
        self.encoder_context = encoder_context
        self.max_priority = max_priority

        self.observation_space = spaces.Box(
            low=-1, high=np.inf, shape=(encoder_context + 1, 5), dtype=np.int32
        )
        self.action_space = spaces.Discrete(max_priority)

        self.reset()

    def _get_info(self):
        return {"info": None}

    def _get_obs(self):
        obs = np.ones((self.encoder_context + 1, 5), dtype=np.int32) * (-1)
        if len(self.processes) > self.data_pointer:
            obs[0, :4] = np.array(self.processes[self.data_pointer])
        for i in range(self.encoder_context):
            if i < len(self.execution_queue.queue):
                obs[i + 1, :4] = self.execution_queue.queue[i][1]  # data
                obs[i + 1, 4] = self.execution_queue.queue[i][0]  # priority
            else:
                break
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options != None:
            try:
                self.data = options["new_data"]
            except:
                print("option not recognized - only 'new_data' implemented")
        self.total_instructions = 0
        self.processes = []
        # print(self.data)
        for pid in range(self.data.shape[0]):
            arrival_time = self.data[pid, 1].astype(np.int32)
            instructions = self.data[pid, 2].astype(np.int32)
            self.total_instructions += instructions
            self.processes.append([pid, arrival_time, instructions, instructions])

        self.current_time = -1
        self.data_pointer = 0
        self.completed_processes = []
        self.current_processes = []
        self.execution_queue = PriorityQueue()

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action):
        # execute as much as possible between last step and now of current process
        # new process arrives
        # assign process priority with action
        # place process into prio queue
        # repeat until action list empty
        if self.data_pointer < len(self.processes):
            # print(self.processes)
            # print(self.data_pointer)
            # print(self.processes[self.data_pointer])
            delta_time = (
                self.processes[self.data_pointer][1] - self.current_time
            ).astype(
                np.int32
            )  # get time difference between last and this step
            # Add next process to current observation, add to priority queue, remove from list of processes
            self.current_processes.append(self.processes[self.data_pointer])
            assign_priority = np.argmax(action)
            # print(assign_priority)
            self.execution_queue.put(
                (assign_priority, (self.processes[self.data_pointer]))
            )
            # print(self.execution_queue.queue)

            # Update current time to arrival time of this process
            self.current_time = self.processes[self.data_pointer][1]
            self.data_pointer += 1  # increment data pointer
        else:
            delta_time = (
                self.total_instructions
            )  # last process added to queue. Finish execution
        # Update highest priority process based on change in time
        for _ in range(delta_time):
            if self.data_pointer >= len(
                self.processes
            ):  # incrementing to finish execution
                self.current_time += 1
            if len(self.current_processes) > 0:
                current_process = self.execution_queue.queue[0]
                remaining_instructions = current_process[1][3]
                remaining_instructions -= 1
                if remaining_instructions == 0:
                    _, proc = self.execution_queue.get()
                    pid = proc[0]
                    # print(self.current_processes)
                    # print(pid)
                    # print(self.current_processes[:])
                    working_index = [p[0] for p in self.current_processes].index(pid)
                    # print(working_index)
                    turnaround_time = self.current_time - proc[1]
                    self.completed_processes.append((pid, turnaround_time))
                    self.current_processes.pop(working_index)
                else:
                    priority = current_process[0]
                    pid = current_process[1][0]
                    arrival = current_process[1][1]
                    instructions = current_process[1][2]
                    self.execution_queue.queue[0] = (
                        priority,
                        [pid, arrival, instructions, remaining_instructions],
                    )
            else:
                break

        # Calculate Reward - 100*(# completed processes) - sum(turnaround time of completed processes)
        reward = 100 * len(self.completed_processes) - sum(
            p[1] for p in self.completed_processes
        )

        # Check if all processes completed
        terminated = (len(self.processes) == len(self.completed_processes)) & (
            len(self.current_processes) == 0
        )

        info = self._get_info()
        obs = self._get_obs()

        return obs, reward, terminated, False, info

    def render(self, mode="human"):
        print(f"Current Time: {self.current_time}")
        print("Running Processes:")
        for priority, process in self.execution_queue.queue:
            print(
                f"  Priority: {priority}, PID: {process[0]}, Arrival Time: {process[1]}, Instructions: {process[2]}, Remaining: {process[3]}"
            )
        if self.execution_queue.not_empty:
            print(f"Current Process: PID {self.execution_queue.queue[0]}")
        print("Completed Processes:")
        for pid, turnaround_time in self.completed_processes:
            print(f"  PID: {pid}, Turnaround Time: {turnaround_time}")
        print()
