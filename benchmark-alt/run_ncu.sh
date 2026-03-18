#!/bin/bash

ncu \
  -o ncu_profile \
  --profile-from-start=off \
  -k regex:compute_ \
  --set=full \
   julia --project $1 
