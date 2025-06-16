#!/bin/bash
clear

start=$(date +%s%3N)

./ORCC.bin main.orcat -o out.ll

end=$(date +%s%3N)

elapsed=$((end - start))
elapsed_s=$((elapsed / 1000))
elapsed_ms=$((elapsed % 1000))

echo "Elapsed time: ${elapsed_s}.${elapsed_ms} seconds"
