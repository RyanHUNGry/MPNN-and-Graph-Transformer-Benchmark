FROM python:3.11-slim

WORKDIR /usr/local/message-pasing-neural-network-vs-graph-transformer-benchmark

# Copy in the source code
COPY . .

RUN pip install --upgrade pip
RUN pip install torch
RUN pip install torch_geometric
RUN pip install scipy numpy

CMD ["sh", "-c", "python run.py"]
