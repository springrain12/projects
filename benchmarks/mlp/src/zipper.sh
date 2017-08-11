#!/bin/bash

option=$1

if [ $# != 1 ]; then
    echo "Usage: $0 <zip or unzip>"; exit 1;
fi

if [ $option == "zip" ]; then
    if [ -d inputs ]; then
        tar cvjf inputs.tar.bz2 inputs; rm -rf inputs
    else
        echo "Error: inputs/ directory does not exist!"; exit 1;
    fi
elif [ $option == "unzip" ]; then
    if [ -f inputs.tar.bz2 ]; then
        rm -rf inputs; tar xf inputs.tar.bz2; rm -f inputs.tar.bz2;
    else
        echo "Error: input.tar.bz2 does not exist!"; exit 1;
    fi
else
    echo "Usage: $0 <zip or unzip>"; exit 1;
fi
