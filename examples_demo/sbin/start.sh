#!/bin/bash
cd ..
# nohup gunicorn --preload --log-level debug --threads 5 -b 0.0.0.0:5050 -t 0 flask_cogkge_demo:app > server.log 2>&1 &
nohup python flask_cogkge_demo.py > server.log 2>&1 &
#nohup uwsgi --socket 127.0.0.1:5050  --protocol=http -p 3 -w flask_cogkge_demo:app > server.log 2>&1 &
#nohup uvicorn --port 5050 flask_cogkge_demo:app > server.log 2>&1 &