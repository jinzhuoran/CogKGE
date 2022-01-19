#!/bin/bash
GIT_OUT=$(git pull)
if [[ $GIT_OUT == *Already* || $GIT_OUT == *已经* ]]
then
    echo $GIT_OUT
else
    cd ../
    pip install -r requirements.txt
    cd -
    ./stop.sh
    ./start.sh
fi
