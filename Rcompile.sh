#!/bin/bash
clear

time bash -c '
  ./ORCC.bin main.orcat -o out.ll --config=ORCC.config &&
  clang -Wno-override-module out.ll stdlib.c -o main &&
  ./main
  echo "Exit code: $?"
'
