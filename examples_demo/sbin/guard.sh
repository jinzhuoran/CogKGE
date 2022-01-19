#!/bin/bash
line_count=$(curl --speed-time 5 --speed-limit 1 127.0.0.1:5050/main.html | wc -l);
if [ $line_count -lt 50 ];
then
    echo "server is stoped, restart it."
    ./restart.sh
else
    echo "server still running."
fi
