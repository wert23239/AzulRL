from threading import Thread
from time import time

# Define 'INFINITY' and 'NEG_INFINITY'
try:
    INFINITY = float("infinity")
    NEG_INFINITY = float("-infinity")
# Windows doesn't support 'float("infinity")'.
except ValueError:
    INFINITY = float(1e3000)       # However, '1e3000' will overflow and return
    NEG_INFINITY = float(-1e3000)  # the magic float Infinity value anyway.


class ContinuousThread(Thread):
    """
    A thread that runs a function continuously,
    with an incrementing 'depth' kwarg, until
    a specified timeout has been exceeded
    """

    def __init__(self, timeout=5, target=None, group=None, name=None, args=(), kwargs={}):
        """
        Store the various values that we use from the constructor args,
        then let the superclass's constructor do its thing
        """
        self._timeout = timeout
        self._target = target
        self._args = args
        self._kwargs = kwargs
        Thread.__init__(self, args=args, kwargs=kwargs,
                        group=group, target=target, name=name)

    def run(self):
        """ Run until the specified time limit has been exceeded """
        depth = 1

        # Times grow exponentially, and we don't want to
        timeout = self._timeout**(1/2.0)
        # start a new depth search when we won't have
        # enough time to finish it

        end_time = time() + timeout

        while time() < end_time:
            self._kwargs['depth'] = depth
            self._most_recent_val = self._target(*self._args, **self._kwargs)
            depth += 1

    def get_most_recent_val(self):
        """ Return the most-recent return value of the thread function """
        try:
            return self._most_recent_val
        except AttributeError:
            print "Error: You ran the search function for so short a time that it couldn't even come up with any answer at all!  Returning a random column choice..."
            import random
            return random.randint(0, 6)


class memoize(object):
    """
    'Memoize' decorator.

    Caches a function's return values,
    so that it needn't compute output for the same input twice.

    Use as follows:
    @memoize
    def my_fn(stuff):
        # Do stuff
    """

    def __init__(self, fn):
        self.fn = fn
        self.memocache = {}

    def __call__(self, *args, **kwargs):
        memokey = (args, tuple(sorted(kwargs.items())))
        if memokey in self.memocache:
            return self.memocache[memokey]
        else:
            val = self.fn(*args, **kwargs)
            self.memocache[memokey] = val
            return val
