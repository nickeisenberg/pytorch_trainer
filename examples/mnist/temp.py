from time import sleep
from tqdm import tqdm

data = range(10)
for ep in range(3):
    pbar = tqdm(data, leave=True)
    for d in pbar:
        sleep(.1)
