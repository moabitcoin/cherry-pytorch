echo "Creating conde environment"
echo conda create -n cherry python=3.6
conda create -n cherry python=3.6
eval "$(conda shell.bash hook)"
echo conda activate cherry
conda activate cherry

echo "https://download.pytorch.org/whl/cpu/torch-1.3.0%2Bcpu-cp36-cp36m-linux_x86_64.whl        \
          --hash=sha256:ce648bb0c6b86dd99a8b5598ae6362a066cca8de69ad089cd206ace3bdec0a5f            \
          \n                                                                                        \
          https://download.pytorch.org/whl/cpu/torchvision-0.4.1%2Bcpu-cp36-cp36m-linux_x86_64.whl  \
          --hash=sha256:593ad12c3c8ce16068566c9eb2bfb39f4834c89a2cc9f0b181e9121b06046b3e            \
          \n" >> requirements_minimal.txt

echo "Installing python packages"
python -m pip install -r requirements_minimal.txt

echo "Installing openAI baseliens"
wget https://github.com/openai/baselines/archive/9ee399f5b20cd70ac0a871927a6cf043b478193f.zip -O baselines.zip && \
    unzip baselines.zip && \
    cd baselines*/ && \
    python -m pip install . && \
    cd ../ && \
    rm -rf baselines*

echo "Reinstalling Pillow"
python -m pip install Pillow==6.2.1
python -m pip install -e .
