# nntopo

Tools for converting neural network layers into graphs and computing simple structural metrics.

## Current script

### `analyze_resnet_block.py`
- Builds a weighted channel graph from a ResNet-18 residual block  
- Compares the pretrained block to an untrained (randomly initialized) version  
- Reports metrics such as:
  - number of nodes and edges  
  - clustering coefficient  
  - average weighted path length  
  - efficiency measures

This is mainly an exploratory script to see whether training affects the connectivity structure implied by the weights.

More experiments may be added later.
