#!/bin/bash
templates=$(python Utils/GenerateTemplateName.py)
for val in $templates; do
    echo running $val
    python RunMultiAgentTest.py --template $val --online_training true --mode within
done