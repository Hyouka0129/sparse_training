# Vit_b_16 86M

consists of 12 encoders

every encoder has two selfattention & a MLP

## test

`python train_transformer.py`

baseline 83.64% 

only updating classifier 72.06%

freeze all attention 81.56%

only updating last encoder 81.96%

only updating last 2 encoders 83.44%

only updating last 1/3 encoders 83.23%

only updating last 3 attention 82.62% 7M

only updating 1/2 attentionhead of 10th encoder, 1/2 attentionhead of 11th encoder, full attention of 12th 83.04% 4.8M

BF16

baseline 81%

only updating last 3 attention 78.9% 
