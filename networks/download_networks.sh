# Make direcotry where the pre-trained networks are stored
mkdir -p pre_trained_networks
# Download Varley's pre-trained network 
wget --no-check-certificate -r 'https://drive.google.com/uc?export=download&id=14nFOcebiAUWopiFkt0BeNsHCQU6rPN7d' -O pre_trained_networks/varley.pth
# Download our mc-dropout pre-trained network
wget --no-check-certificate -r 'https://drive.google.com/uc?export=download&id=1P1iL-82y4sViRreIXn-UmqrCbb3VA3ld' -O pre_trained_networks/mc_dropout_rate_0.2_lat_2000.model
