# -*-coding:utf-8 -*-
"""
@File    :   timer.py
@Time    :   2024/10/02 13:11:59
@Author  :   Urvan Christen
@Version :   1.0
@Contact :   urvan.christen@gmail.com
@Desc    :   Timer utility
"""


import time

MASK = "%duration"


class Timer:
    def __init__(self, message=None):
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
        self.message = message if message else "Elapsed time: %duration"
        if MASK not in self.message:
            self.message += f" {MASK}"

    def __enter__(self):
        print(self.message.replace(MASK, "..."), end="\r")
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        print(self.message.replace(MASK, self.format_time(self.elapsed_time)))

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time

    def get_elapsed_time(self):
        return self.elapsed_time

    def reset(self):
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None

    @staticmethod
    def format_time(time):
        if time < 1e-6:
            return f"{time * 1e9:.2f} ns"
        elif time < 1e-3:
            return f"{time * 1e6:.2f} µs"
        elif time < 1:
            return f"{time * 1e3:.2f} ms"

        return f"{time:.3f} seconds"
