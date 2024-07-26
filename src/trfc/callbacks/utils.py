import os

def rank_zero_only(func):
    def wrapper(*args, **kwargs):
        rank = os.getenv("RANK", 0)
        if int(rank) == 0:
            func(*args, **kwargs)
    return wrapper
