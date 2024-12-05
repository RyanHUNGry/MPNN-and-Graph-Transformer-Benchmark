# Message Passing Neural Network VS Graph Transformer Benchmark
## Aim
This benchmark evaluates message passing neural network performance against on common graph neural network benchmark datasets, and on the Long Range Graph Benchmark [Dwivedi et al. 2022](https://arxiv.org/abs/2206.08164).

Our benchmark supports the models GCN, GAT, GIN, and GraphGPS. The datasets supported are IMDB-BINARY, Cora, Enzymes, and PascalVOC-SP.

## Run
The benchmark is PyTorch based and leverages PyTorch Geometric for GNN operations.

We supply a Docker image to run our benchmark. The output of `run.py` will be serialized to disk at `outputs/results.json`. A running execution log will also be available via `stdout`.

> **⚠️ Notice:** Please expect a longer initial image pull. We find that baking the raw data into the image is faster than loading and processing at runtime. Training times can also vary given your machine's allocated CPU and RAM. GraphGPS, the transformer model, may take the longest due to its multi-head, global attention mechanism. Train and test accuracies can slightly differ as well, since our implementation shuffles datasets prior to splits.

```
docker pull ghcr.io/ryanhungry/gnn-benchmark-image:latest
docker run --name my-local-benchmark ghcr.io/ryanhungry/gnn-benchmark-image:latest
```

To see `outputs/results.json` once the container finishes running the model, copy the file from the exited container to your local filesystem.

```
docker cp <container_id>:/usr/local/message-pasing-neural-network-vs-graph-transformer-benchmark/outputs/results.json <local_path>
```

## Configuration
We supply a `params.json` configuration file to pass in model parameters, organized by model and dataset type. The base hyperparameter configuration follows our setup in our report, so the results can be reproduced without additional configuration.

To supply your own custom set of hyperparameters, clone this repository and edit `params.json` locally. Ensure field names are not changed, and that only JSON integer field types are used. Then, mount your `params.json` as a volume into the container during runtime.

```
docker run -v <local_path>/params.json:/params.json message-passing-neural-network-vs-graph-transformer-benchmark:latest
```

