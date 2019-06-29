for fl in ./*; do wget -nc -t 3 -i $fl/Links.txt -P $fl; done
