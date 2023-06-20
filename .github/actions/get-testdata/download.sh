# args: outputfolder, filename, url
mkdir -p $1
cd $1
wget -O $2 $3
# extract as needed
if  [[ $2 == *.tar.gz ]] ;
then
    tar -xf $2
fi
