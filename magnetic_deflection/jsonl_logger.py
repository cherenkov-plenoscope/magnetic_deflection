import logging

def init(path):
    l = logging.Logger(name=path)
    file_handler = logging.FileHandler(filename=path, mode="w")
    datefmt_iso8601 = "%Y-%m-%dT%H:%M:%S"
    formatter = logging.Formatter(
        fmt='{"t":"%(asctime)s", "m":"%(message)s"}', datefmt=datefmt_iso8601,
    )
    file_handler.setFormatter(formatter)
    l.addHandler(file_handler)
    l.setLevel(logging.DEBUG)
    return l
