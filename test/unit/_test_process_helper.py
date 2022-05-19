"""
Helper script for testing signal handling

- If it receives SIGTERM, immediately exit "21"
- If it doesn't receive a signal, sleep for 3 seconds then exit "-1"
"""

import signal
import time


def signal_handler(signalnum, *_):
    assert signalnum == signal.SIGTERM
    exit(21)


def main():
    signal.signal(signal.SIGTERM, signal_handler)
    time.sleep(3)
    exit(-1)


if __name__ == "__main__":
    main()
