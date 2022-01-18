#!/bin/bash
pid=$(ps -aux | grep 'flask_cogkge_demo' | grep -v "grep" |awk '{print $2}')
kill -9 $pid
