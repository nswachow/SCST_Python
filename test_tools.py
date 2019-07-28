from typing import Callable


def exceptionTest(test_func: Callable, except_type: Exception):

    was_thrown = False
    try:
        test_func()
    except except_type:
        was_thrown = True

    assert was_thrown
