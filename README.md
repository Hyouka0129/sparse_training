# Sparse_training

## Usage

```bash
$ cd sparse_training
$ python train.py
```

## Modify sparsity scheme

Modify `sparsity_ratio` in `model_generator.py`

## test

baseline 82.92% 

sparsity_ratio=[0.25,0.25,0.25,0.25,0.25,0.25] 82.79% saving 52% memory

sparsity_ratio=[0.5,0.5,0.5,0.5,0.5,0.5] 79.65%

sparsity_ratio=[0.25,0.5,0.25,0.5,0.25,0.5] 80.86%

sparsity_ratio=[0.5,0.25,0.5,0.25,0.5,0.25] 80.40%

