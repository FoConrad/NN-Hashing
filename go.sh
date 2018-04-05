#!/usr/bin/env bash

###############################################################################
#
# This code continually finds collisions for target_random then saves the 
# binarized output file. It then checks if the hash for the binarized output
# file matches the hash for the target_random file. If so, a true collision has
# been found.
#
###############################################################################

target_string=$(./hashing.py test_files/target_random | cut -d' ' -f1)

echo "$target_string -- target"
while true; do
    ./hashing.py test_files/target_random --source <(head -22 /dev/urandom) --output nill --iters 3000 --loss-reg 0.25 --cuda > /dev/null
    attacked_hash=$(./hashing.py nill | cut -d' ' -f1)
    echo -ne "$attacked_hash -- attack\r"
    if [[ "$target_string" == "$attacked_hash" ]]; then
        echo -e "\nDone! Collision is file nill"
        break
    fi
done
