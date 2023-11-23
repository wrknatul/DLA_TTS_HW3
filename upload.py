import gdown
url = "https://drive.google.com/u/0/uc?id=1-EdH0t0loc6vPiuVtXdhsDtzygWNSNZx"
output = "train.txt"
gdown.download(url, output, quiet=False)