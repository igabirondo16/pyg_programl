import time
from datetime import timedelta
from statistics import mean, stdev


def measure_execution_time(n: int):
    """Example decorator for measuring the execution time
    of n runs.
    It compute the mean and the standard deviation of the
    n runs.

    Args:
        n (int): Number of runs to perform.
    """

    def decorator(func):
        def wrapper():
            times = []

            for _ in range(n):
                start_time = time.monotonic()
                func()
                end_time = time.monotonic()

                exec_time = end_time - start_time
                times.append(exec_time)

            mean_exec_time = mean(times)
            stdev_exec_time = stdev(times)

            print(
                f"Mean execution time of {n} runs: {timedelta(seconds=mean_exec_time)}"
            )
            print(
                f"Standard deviation of the execution time of {n} runs: {timedelta(seconds=stdev_exec_time)}"
            )

        return wrapper

    return decorator
