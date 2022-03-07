echo "$(dirname "$0")"
cd "$(dirname "$0")"

source /home/cbyrohl/anaconda3/bin/activate root
conda activate darepo39

make html

# copy temporary copy from trusted client to cbyrohl.de
rsync -arv build/html/* hetzner:/home/cbyrohl/public_content/astrodask
