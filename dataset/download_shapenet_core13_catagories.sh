SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`

cd $SCRIPTPATH/
wget https://cloud.enpc.fr/s/j2ECcKleA1IKNzk/download
mkdir -p shapenet_core13
mv download shapenet_core13/
cp shapenet_core13_synsetoffset2category.txt shapenet_core13/
cd $SCRIPTPATH/shapenet_core13
unzip download
rm download
