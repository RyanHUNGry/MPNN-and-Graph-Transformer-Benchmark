# Message Passing Neural Network VS Graph Transformer Benchmark
## Aim
This benchmark evaluates message passing neural network performance against on common graph neural network benchmark datasets, and on the Long Range Graph Benchmark [Dwivedi et al. 2022](https://arxiv.org/abs/2206.08164).

Our benchmark supports the models GCN, GAT, GIN, and GraphGPS. The datasets supported are IMDB-BINARY, Cora, Enzymes, and PascalVOC-SP.

For formal result discussion and analysis, see [paper](https://production-gradescope-uploads.s3-us-west-2.amazonaws.com/uploads/pdf_attachment/file/183055548/DSC_Capstone_Project_1.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAV45MPIOWYLR224KG%2F20241205%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20241205T021152Z&X-Amz-Expires=10800&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEFEaCXVzLXdlc3QtMiJGMEQCIBSPGKqYYP0VW6a2K6qLDpY%2Bjm8dvNR3fvO0if%2FL%2Bx%2FEAiABJQknqf84OTPuvGZmdlnhnYVY%2Fk0Bs6erkZD9OZQcrCrDBQj6%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAAaDDQwNTY5OTI0OTA2OSIMVFaM61%2FEYCde7tXsKpcFv5aNA%2ByOIzNXU%2Fjes52wHat8BvGvq9B3ausW%2FIzhiqX1ZmbQl27QKVAhWGhl55q%2BpDgrP7juukCy%2Bml6vKxLco8wKtDNUb3en%2B1vZQTb7BL5vd0WpbmNzkf3Hmhak1Ds0AER3vmE%2BZGpEQV%2F9GI1YzYv3RXOBgCQ5wkHkMHqCiG8flVV%2BRq1WiLAz9222zu40HOCaN1SSWz3l4TkITP90pRz7WBUPgxAbeqvGbkFG6R5SVqphr6wO4uI81QtNvZMDghiCxeqwo6dGwkBI6J%2BJWPnAeeg6deQ6aC7dZD2u46ZZthPRewlo1l7CCDjNEjSjUex2BkXTj70qSXgsDh0eFi47i40NtnoDPZw8Vb7SK1PHVC9ENKF0wIEeRNv0EBSzn6jIoR9dgx8BSzwA19Ih4pTQ7Q6V3fV%2FzXXSePhYB25Ce%2FJHc1MqpzLZFtmaMYudOiJLAW%2B9LnAfiutLfrvzX54GZxD5mWaZB8nbDGgbn4eXeaejuBsyWSFtpPW895%2FF98Q%2BTT2kvpYkKwx9tSknCOa1thwCHDLxVI1EKXTEJWWzWpX4TbKAkmvyZzILxoEcE9Pk9GlAAbmVHobr2ijXWgnhNdKyC4tX7wA1YQFf9qROKA0Tcbx7PueDMxx65rmQJv1FtFFaklOsytRm5zXF0Xn6Rfo5x%2BXgYw3ztRilp6L6zbLTEW2ZYAr9DKHJqV33nXTuOmV9sVpizy1gTXuT8GAddx6AoTogViSx1fhUtzJR0I9uVnrNq4Xfkhfd3uaJbo3MYcMIcbdQqOgU1zbvCqsNiuQynlrkgNToWSpClYfQqB07I0Jc5lPNM6%2FktIPHEKpN2KrMKTCIK8Bh8MhF0HqVOkB2rY0VxUcjloGtvjOObIqHcWfMOH2w7oGOrIBIvv7Fo1QS%2F%2FdqQe60mQSkJcLQuHIJOVNi%2BmEz%2F%2FWefs8CdwHG%2F%2Fellh6CBuDYEskK2GL%2F1Se8txffRCNYczdc5LxMWyc0bTD%2BqtgAXw14tjnfX5h%2BH2aNZ8DjAjzsChJu%2FOXfPJbqU7KtIbGzkBqnkWkzUSyYraDGbXojzohaeyZPKS0Yz%2F%2BdrIuxrvZvg7g%2B4t%2BoGeQeovXvpjh3JROzvXYH5SjUMc%2FMQXSA5v99wfzEg%3D%3D&X-Amz-SignedHeaders=host&X-Amz-Signature=3323e2ed790c5abafcd8ea0015d480ef1cbb5cfa1cadf4947b678f183966fe75).

## Run
The benchmark is PyTorch based and leverages PyTorch Geometric for GNN operations.

We supply a Docker image to run our benchmark. The output of `run.py` will be serialized to disk at `outputs/results.json`. A running execution log will also be available via `stdout`.

> **⚠️ Notice:** Please expect a longer initial image pull. We find that baking the raw data into the image is faster than loading and processing at runtime. Training times can also vary given your machine's allocated CPU and RAM. GraphGPS, the transformer model, may take the longest due to its multi-head, global attention mechanism. Train and test accuracies can slightly differ as well, since our implementation shuffles datasets prior to splits.

```
docker pull ghcr.io/ryanhungry/message-passing-neural-network-vs-graph-transformer-benchmark:latest
docker run --name my-local-benchmark ghcr.io/ryanhungry/message-passing-neural-network-vs-graph-transformer-benchmark:latest
```

To see `outputs/results.json` once the container finishes running the model, copy the file from the exited container to your local filesystem.

```
docker cp <container_id>:/usr/local/message-pasing-neural-network-vs-graph-transformer-benchmark/outputs/results.json <local_path>
```

## Configuration
We supply a `params.json` configuration file to pass in model parameters, organized by model and dataset type. The base hyperparameter configuration follows our setup in our [paper](https://production-gradescope-uploads.s3-us-west-2.amazonaws.com/uploads/pdf_attachment/file/183055548/DSC_Capstone_Project_1.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIAV45MPIOWYLR224KG%2F20241205%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20241205T021152Z&X-Amz-Expires=10800&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEFEaCXVzLXdlc3QtMiJGMEQCIBSPGKqYYP0VW6a2K6qLDpY%2Bjm8dvNR3fvO0if%2FL%2Bx%2FEAiABJQknqf84OTPuvGZmdlnhnYVY%2Fk0Bs6erkZD9OZQcrCrDBQj6%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAAaDDQwNTY5OTI0OTA2OSIMVFaM61%2FEYCde7tXsKpcFv5aNA%2ByOIzNXU%2Fjes52wHat8BvGvq9B3ausW%2FIzhiqX1ZmbQl27QKVAhWGhl55q%2BpDgrP7juukCy%2Bml6vKxLco8wKtDNUb3en%2B1vZQTb7BL5vd0WpbmNzkf3Hmhak1Ds0AER3vmE%2BZGpEQV%2F9GI1YzYv3RXOBgCQ5wkHkMHqCiG8flVV%2BRq1WiLAz9222zu40HOCaN1SSWz3l4TkITP90pRz7WBUPgxAbeqvGbkFG6R5SVqphr6wO4uI81QtNvZMDghiCxeqwo6dGwkBI6J%2BJWPnAeeg6deQ6aC7dZD2u46ZZthPRewlo1l7CCDjNEjSjUex2BkXTj70qSXgsDh0eFi47i40NtnoDPZw8Vb7SK1PHVC9ENKF0wIEeRNv0EBSzn6jIoR9dgx8BSzwA19Ih4pTQ7Q6V3fV%2FzXXSePhYB25Ce%2FJHc1MqpzLZFtmaMYudOiJLAW%2B9LnAfiutLfrvzX54GZxD5mWaZB8nbDGgbn4eXeaejuBsyWSFtpPW895%2FF98Q%2BTT2kvpYkKwx9tSknCOa1thwCHDLxVI1EKXTEJWWzWpX4TbKAkmvyZzILxoEcE9Pk9GlAAbmVHobr2ijXWgnhNdKyC4tX7wA1YQFf9qROKA0Tcbx7PueDMxx65rmQJv1FtFFaklOsytRm5zXF0Xn6Rfo5x%2BXgYw3ztRilp6L6zbLTEW2ZYAr9DKHJqV33nXTuOmV9sVpizy1gTXuT8GAddx6AoTogViSx1fhUtzJR0I9uVnrNq4Xfkhfd3uaJbo3MYcMIcbdQqOgU1zbvCqsNiuQynlrkgNToWSpClYfQqB07I0Jc5lPNM6%2FktIPHEKpN2KrMKTCIK8Bh8MhF0HqVOkB2rY0VxUcjloGtvjOObIqHcWfMOH2w7oGOrIBIvv7Fo1QS%2F%2FdqQe60mQSkJcLQuHIJOVNi%2BmEz%2F%2FWefs8CdwHG%2F%2Fellh6CBuDYEskK2GL%2F1Se8txffRCNYczdc5LxMWyc0bTD%2BqtgAXw14tjnfX5h%2BH2aNZ8DjAjzsChJu%2FOXfPJbqU7KtIbGzkBqnkWkzUSyYraDGbXojzohaeyZPKS0Yz%2F%2BdrIuxrvZvg7g%2B4t%2BoGeQeovXvpjh3JROzvXYH5SjUMc%2FMQXSA5v99wfzEg%3D%3D&X-Amz-SignedHeaders=host&X-Amz-Signature=3323e2ed790c5abafcd8ea0015d480ef1cbb5cfa1cadf4947b678f183966fe75), so the results can be reproduced without additional configuration.

To supply your own custom set of hyperparameters, clone this repository and edit `params.json` locally. Ensure field names are not changed, and that only JSON integer field types are used. Then, mount your `params.json` as a volume into the container during runtime.

```
docker run -v <local_path>/params.json:/params.json message-passing-neural-network-vs-graph-transformer-benchmark:latest
```

