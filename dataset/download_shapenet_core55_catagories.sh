SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`

cd $SCRIPTPATH
gdown https://drive.google.com/uc?id=1sJd5bdCg9eOo3-FYtchUVlwDgpVdsbXB
mkdir -p shapenet_core55
mv shapenet57448xyzonly.npz shapenet_core55/
