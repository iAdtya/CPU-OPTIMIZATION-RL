from abc import ABC, abstractmethod
import numpy as np
import time

class Scheduler(ABC):
    def __init__(self, data):
        self.gantt = []
        self.data = data
        if data is not None:
            self.pids = data[:,0].astype(int)
            self.arrivals = data[:,1]
            self.instr_count = data[:,2]

    def cpu_util(self):
        "Ranges from 0-1, can be converted to percentage"
        cpu_utilization = (len(self.gantt) - self.gantt.count(-1)) / float(len(self.gantt))
        return cpu_utilization

    def throughput(self):
        """Number of processes completed per time unit"""
        #print("data size", self.data.shape[0])
        return len(self.pids) / len(self.gantt)

    def turnaround_time(self):
        """Average turnaround time - time between submission to completion"""
        turnaround_times = []
        for pid in self.pids:
            first_index = self.gantt.index(pid)
            last_index = len(self.gantt) - 1 - self.gantt[::-1].index(pid)
            turnaround_times.append(last_index - first_index)
        return sum(turnaround_times) / len(turnaround_times)

    def waiting_time(self):
        "Average waiting time - amount of time a process has been waiting in the ready queue not including execution and I/O"
        waiting_times = []
        for pid in self.pids:
            arrival_time = self.arrivals[pid]
            instruction_count = self.arrivals[pid]
            finish_time = len(self.gantt) - 1 - self.gantt[::-1].index(pid)
            waiting_times.append(finish_time - arrival_time - instruction_count)
        return sum(waiting_times) / len(waiting_times)

    def response_time(self):
        "Average response time - amount of time it takes from when a request was submitted until the first response is produced"
        response_times = []
        for pid in self.pids:
            arrival_time = self.arrivals[pid]
            first_index = self.gantt.index(pid)
            response_times.append(first_index - arrival_time)
        return sum(response_times) / len(response_times)
    
    def time_run(self):
        "Overhead - amount of time (real) to complete running a particular dataset"
        start_time = time.time()
        self.run()
        stop_time = time.time()
        self.stat_runtime = stop_time - start_time

    def calc_stats(self):
        self.stat_cpu_util = self.cpu_util()
        self.stat_throughput = self.throughput()
        self.stat_turnaround_time = self.turnaround_time()
        self.stat_response_time = self.response_time()
        self.stat_waiting_time = -self.waiting_time()
        self.stat_mean_runtime = self.stat_runtime / len(self.pids)

    def print_stats(self):
        print("CPU Utilization :",self.stat_cpu_util)
        print("Throughput      :",self.stat_throughput)
        print("Turnaround Time :",self.stat_turnaround_time)
        print("Waiting Time    :",self.stat_waiting_time)
        print("Response Time   :",self.stat_response_time)
        print("Runtime         :",self.stat_runtime)
        print("Mean Runtime    :",self.stat_mean_runtime)

    @abstractmethod
    def run():
        pass