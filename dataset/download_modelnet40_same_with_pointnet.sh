SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`

cd $SCRIPTPATH/
wget https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip --no-check-certificate
mkdir -p modelnet40_from_pointnet
mv modelnet40_ply_hdf5_2048.zip modelnet40/
cd $SCRIPTPATH/modelnet40/
unzip modelnet40_ply_hdf5_2048.zip
rm modelnet40_ply_hdf5_2048.zip
