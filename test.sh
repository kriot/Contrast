#!/bin/bash

imgs="01 04 29 44 52 53 56"

rm -rf ContrastImages
mkdir ContrastImages

for img in $imgs
do
	./contrast ./CropedImages/$img.tif
	mv output.tif ./ContrastImages/$img.tif
	echo "$img is done" 
done
echo "All is done. See result in ./ContrastImages"
