#!/bin/bash
templates=$(python Utils/GenerateTemplateName.py)
for val in $templates; do
    echo running $val
    python MultiAgentTestOnly.py --template $val --online_training false
done