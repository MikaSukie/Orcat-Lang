#!/bin/bash
clear
  ./ORCC.bin main.orcat -o out.ll &&
  opt -O2 out.ll -o out.opt.ll &&
  clang -O2 -Wno-override-module out.opt.ll stdlib.c -o main &&
time bash -c '  ./main
  echo "Exit code: $?"
'
