# imagenet
CUDA_VISIBLE_DEVICES=1 poetry run python scripts/bimodal_data_align.py dataset=imagenet imagenet.sim_dim=10
CUDA_VISIBLE_DEVICES=1 poetry run python scripts/bimodal_data_align.py dataset=imagenet imagenet.sim_dim=25
CUDA_VISIBLE_DEVICES=1 poetry run python scripts/bimodal_data_align.py dataset=imagenet imagenet.sim_dim=50
CUDA_VISIBLE_DEVICES=1 poetry run python scripts/bimodal_data_align.py dataset=imagenet imagenet.sim_dim=100
CUDA_VISIBLE_DEVICES=1 poetry run python scripts/bimodal_data_align.py dataset=imagenet imagenet.sim_dim=150
CUDA_VISIBLE_DEVICES=1 poetry run python scripts/bimodal_data_align.py dataset=imagenet imagenet.sim_dim=200
CUDA_VISIBLE_DEVICES=1 poetry run python scripts/bimodal_data_align.py dataset=imagenet imagenet.sim_dim=500
CUDA_VISIBLE_DEVICES=1 poetry run python scripts/bimodal_data_align.py dataset=imagenet train_test_ratio=0.1 imagenet.sim_dim=10
CUDA_VISIBLE_DEVICES=1 poetry run python scripts/bimodal_data_align.py dataset=imagenet train_test_ratio=0.1 imagenet.sim_dim=25
CUDA_VISIBLE_DEVICES=1 poetry run python scripts/bimodal_data_align.py dataset=imagenet train_test_ratio=0.1 imagenet.sim_dim=50
CUDA_VISIBLE_DEVICES=1 poetry run python scripts/bimodal_data_align.py dataset=imagenet train_test_ratio=0.1 imagenet.sim_dim=100
CUDA_VISIBLE_DEVICES=1 poetry run python scripts/bimodal_data_align.py dataset=imagenet train_test_ratio=0.1 imagenet.sim_dim=150
CUDA_VISIBLE_DEVICES=1 poetry run python scripts/bimodal_data_align.py dataset=imagenet train_test_ratio=0.1 imagenet.sim_dim=200
CUDA_VISIBLE_DEVICES=1 poetry run python scripts/bimodal_data_align.py dataset=imagenet train_test_ratio=0.1 imagenet.sim_dim=500
