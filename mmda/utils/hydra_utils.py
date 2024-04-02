import functools

import hydra


def hydra_main(*args, **kw):  # noqa: D103

    main = hydra.main(*args, **kw)

    def main_decorator(f):
        returned_values = []

        @functools.wraps(f)
        def f_wrapper(*args, **kw):
            ret = f(*args, **kw)
            returned_values.append(ret)
            return ret

        wrapped = main(f_wrapper)

        @functools.wraps(wrapped)
        def main_wrapper(*args, **kw):
            wrapped(*args, **kw)
            return returned_values[0] if len(returned_values) == 1 else returned_values

        return main_wrapper

    return main_decorator
