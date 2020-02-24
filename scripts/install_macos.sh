echo "Installing python packages"
pip install -r requirements_minimal.txt

echo "Pytorch installation"
conda install pytorch torchvision -c pytorch

echo "Installing openAI baseliens"
wget https://github.com/openai/baselines/archive/9ee399f5b20cd70ac0a871927a6cf043b478193f.zip -O baselines.zip && \
    unzip baselines.zip && \
    cd baselines*/ && \
    python -m pip install . && \
    cd ../ && \
    rm -rf baselines*

echo "Reinstalling Pillow"
pip install Pillow==6.2.1