#!/bin/bash
clear

time bash -c '
  python ORCC.py main.orcat -o out.ll &&
  opt -O2 out.ll -o out.opt.ll &&
  clang -O2 -Wno-override-module out.opt.ll stdlib.c -o main &&
  ./main
  echo "Exit code: $?"
'
