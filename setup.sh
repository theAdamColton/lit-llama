python -m venv venv
source venv/bin/activate

apt install vim
pip install -r requirements.txt
pip install bpython
python scripts/download.py --repo_id decapoda-research/llama-7b-hf --local_dir checkpoints/hf-llama/7B
python scripts/convert_hf_checkpoint.py --model_size 7B
