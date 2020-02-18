# Make direcotry where the pre-trained networks are stored
mkdir -p pre_trained_networks
# Download the pre-trained networks network 
wget --no-check-certificate -r 'https://drive.google.com/uc?export=download&id=1s4z00OFRqA1VK1PX9gEt0SCG6y4TK3CJ' -O trained_networks.zip
unzip -q trained_networks.zip -d pre_trained_networks
rm trained_networks.zip
