for fl in ./*; do wget -tries=3 -i $fl/Links.txt -P $fl; done
