# install necessary packages
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y build-essential git curl wget python3-pip

# Python venv
python3 --version
python3 -m venv venv
souce venv/bin/acitvate

# upgrade pip
pip install --upgrade pip

# install DeepSeek R1 model
git clone https://github.com/deepseek-ai/DeepSeek-R1.git
