# classification
CUDA_VISIBLE_DEVICES=1 poetry run python mmda/bimodal_classification.py dataset=handwriting handwriting.sim_dim=10
CUDA_VISIBLE_DEVICES=1 poetry run python mmda/bimodal_classification.py dataset=handwriting handwriting.sim_dim=25
CUDA_VISIBLE_DEVICES=1 poetry run python mmda/bimodal_classification.py dataset=handwriting handwriting.sim_dim=50
CUDA_VISIBLE_DEVICES=1 poetry run python mmda/bimodal_classification.py dataset=handwriting handwriting.sim_dim=100
CUDA_VISIBLE_DEVICES=1 poetry run python mmda/bimodal_classification.py dataset=handwriting handwriting.sim_dim=200
CUDA_VISIBLE_DEVICES=1 poetry run python mmda/bimodal_classification.py dataset=handwriting handwriting.sim_dim=500
