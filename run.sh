#!/bin/sh

#Run API
export FLASK_APP=src/api/app.py
API_PID_FILE=api_pid.txt
if test -f "$API_PID_FILE";
then
    pid=`cat "$API_PID_FILE"`
    echo $pid
    if [ "$pid" != '' ]
    then
        kill -9 $pid
        sleep 1
        rm "$API_PID_FILE"
        sleep 1
    fi
    echo "RUNNING INFERENCE API"
    nohup sh -c 'flask run' 2>&1 > api.log &
    sleep 1
    pgrep flask > "$API_PID_FILE"
fi

#Run Streamlit
STREAMLIT_PID_FILE=streamlit_pid.txt
if test -f "$STREAMLIT_PID_FILE";
then
    pid=`cat "$STREAMLIT_PID_FILE"`
    echo $pid
    if [ "$pid" != '' ]
    then
        kill -9 $pid
        sleep 1
        rm -f "$STREAMLIT_PID_FILE"
        sleep 1
    fi
    echo "RUNNING STREAMLIT DASHBOARD"
    nohup sh -c 'streamlit run --server.port 8000 dashboard.py' 2>&1 > streamlit.log &
    
    sleep 1
    pgrep streamlit > "$STREAMLIT_PID_FILE"
fi