#!/bin/bash

# Pull data
URL="http://35.184.200.199/data_v1"
wget $URL -O raw_data.zip

# Unzip data
unzip raw_data.zip

# Remove zip
<<<<<<< HEAD
rm raw_data.zip
=======
#rm raw_data.zip
>>>>>>> d98003a619425ee5754ec7fe6e949238a741af66
