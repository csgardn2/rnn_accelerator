#!/bin/bash
# The system will freeze if python tries to use too much memory
# (which happens alot since python is really wasteful).  Laucnh a terminal
# which creashes gracefully if more than 12GB of cumulative memory are consumed

ulimit -v 1048576

while true
do
    echo "Launching sub-terminal with 12GB virtual memory limit"
    bash && break
    echo "12GB memory limit reached, closing all running programs"
done

echo "Exiting limited mode.  You now have free-reign of all memory"
