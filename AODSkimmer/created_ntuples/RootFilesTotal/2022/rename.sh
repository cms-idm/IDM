#!/usr/bin/env bash

for f in Signal_Signal_Mchi-*.root_output.root; do
    new="${f#Signal_}"                  # Remove the first "Signal_"
    new="${new/.root_output.root/_output.root}"   # Fix the suffix
    echo "$f -> $new"
    mv "$f" "$new"
done