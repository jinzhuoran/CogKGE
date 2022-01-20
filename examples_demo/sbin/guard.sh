#!/bin/bash
pid=$(ps -aux | grep 'flask_cogkge_demo' | grep -v "grep" |awk '{print $2}')
if [ -z "$pid" ];
then
    echo "server is stoped, restart it."
    ./restart.sh
else
    echo "server still running."
fi
