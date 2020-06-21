#!/bin/bash

# Pull data
URL="http://35.184.200.199/data_v1"
wget $URL -O raw_data.zip

# Unzip data
unzip raw_data.zip

# Remove zip
rm raw_data.zip
