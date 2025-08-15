#!/bin/bash
clear

time bash -c '
  ./ORCC.bin main.orcat -o out.ll &&
  clang -Wno-override-module out.ll stdlib.c -o main -lpthread
'
