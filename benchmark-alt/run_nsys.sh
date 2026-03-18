#!/bin/bash

LD_LIBRARY_PATH=$(julia -e 'println(joinpath(Sys.BINDIR, Base.LIBDIR, "julia"))'):$LD_LIBRARY_PATH

nsys  profile  \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --gpu-metrics-device=all \
  julia --project $1 
