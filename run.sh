#!/bin/bash
sudo docker run -it --rm \
  --name HyperelasticSolver \
  --mount type=volume,source=julia_docker,target=/root/.julia \
  --mount type=bind,source="$(pwd)",target=/app \
  --workdir /app \
  julia:1.9 julia -t 8 main.jl
