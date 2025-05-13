#!/bin/bash
sudo docker run -it --rm -v ~/Documents/GitHub/HyperelasticSolver:/app \
  -v julia_docker:/root/.julia \
  -w /app julia:1.9 julia -t 8 main.jl
