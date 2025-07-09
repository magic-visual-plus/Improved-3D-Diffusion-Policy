import time
from loguru import logger

# 用于debug代码执行时间
time_map = ['s', 'ms', 'us', 'ns']


def format_time(time_sec):
    for i in range(len(time_map)):
        if time_sec > 1:
            break
        time_sec *= 1000
    res = f'{round(time_sec, 1)}{time_map[i]}'
    return res


class Dbg_Timer(object):
    def __enter__(self):
        self.start = time.time()
        return self

    def __init__(self, tag, time_th=0.001, note=''):
        self.start = time.time()
        self.tag = tag
        self.time_th = time_th
        self.note = note

    def _flied(self):
        return time.time() - self.start

    def flied(self):
        return format_time(self._flied())

    def timeout(self, time_th=None):
        time_th = time_th or self.time_th or 0
        return self._flied() > time_th

    def __exit__(self, *args):
        if self.timeout():
            _tag = f'{self.tag}-{self.note}' if self.note else self.tag
            logger.info(f'[{_tag}]: Exec Time={self.flied()}')