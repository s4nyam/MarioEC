#!/usr/bin/env bash

#ls -1tp | grep -E "neat-checkpoint-parallel-[0-9]" | tail -n +5 | xargs -I {} rm -- {}
ls -1tp | grep -E "neat-checkpoint-[0-9]" | tail -n +5 | xargs -I {} rm -- {}
