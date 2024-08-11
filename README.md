A lightweight trainer that features callbacks and DDP support written with only
615 lines of core code.

Easily
  * log intra batch values to a csv as well as a tqdm progress bar,
  * save the weights of a model based on an increase or decrease of a logged metric,
  * train on multiple GPUs and multiple nodes with `DataDistributedParallel`,
  * write any custom feature you would like with the use of a callback.
