import os

def rank_zero_only(func):
    rank = os.getenv("RANK", 0)
    def wrapper(*args, **kwargs):
        if int(rank) == 0:
            func(*args, **kwargs)
    return wrapper
