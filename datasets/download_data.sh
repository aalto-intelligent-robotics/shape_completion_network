# Download and unzip ground-truth meshes
wget --no-check-certificate -r 'https://drive.google.com/uc?export=download&id=1SGyNKkptHXaBABnADUiTom_Yt2ghtRMg' -O ground_truth_meshes.zip 
unzip -q ground_truth_meshes.zip
rm ground_truth_meshes.zip
# Download and unzip training data
wget --no-check-certificate -r 'https://drive.google.com/uc?export=download&id=1Op7Teux9pw5YkI9uS_nn7Nzz_xNkPnkd' -O training_data.zip 
unzip -q training_data.zip
rm training_data.zip
