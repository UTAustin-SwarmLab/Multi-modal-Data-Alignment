# pitts
CUDA_VISIBLE_DEVICES=6 poetry run python scripts/bimodal_shuffled_data.py dataset=pitts pitts.sim_dim=10
CUDA_VISIBLE_DEVICES=6 poetry run python scripts/bimodal_shuffled_data.py dataset=pitts pitts.sim_dim=50
CUDA_VISIBLE_DEVICES=6 poetry run python scripts/bimodal_shuffled_data.py dataset=pitts pitts.sim_dim=100
CUDA_VISIBLE_DEVICES=6 poetry run python scripts/bimodal_shuffled_data.py dataset=pitts pitts.sim_dim=200
CUDA_VISIBLE_DEVICES=6 poetry run python scripts/bimodal_shuffled_data.py dataset=pitts pitts.sim_dim=500
CUDA_VISIBLE_DEVICES=6 poetry run python scripts/bimodal_shuffled_data.py dataset=pitts pitts.sim_dim=1000
CUDA_VISIBLE_DEVICES=6 poetry run python scripts/bimodal_shuffled_data.py dataset=pitts pitts.sim_dim=2000
CUDA_VISIBLE_DEVICES=6 poetry run python scripts/bimodal_shuffled_data.py dataset=pitts train_test_ratio=0.1 pitts.sim_dim=10
CUDA_VISIBLE_DEVICES=6 poetry run python scripts/bimodal_shuffled_data.py dataset=pitts train_test_ratio=0.1 pitts.sim_dim=50
CUDA_VISIBLE_DEVICES=6 poetry run python scripts/bimodal_shuffled_data.py dataset=pitts train_test_ratio=0.1 pitts.sim_dim=100
CUDA_VISIBLE_DEVICES=6 poetry run python scripts/bimodal_shuffled_data.py dataset=pitts train_test_ratio=0.1 pitts.sim_dim=200
CUDA_VISIBLE_DEVICES=6 poetry run python scripts/bimodal_shuffled_data.py dataset=pitts train_test_ratio=0.1 pitts.sim_dim=500
CUDA_VISIBLE_DEVICES=6 poetry run python scripts/bimodal_shuffled_data.py dataset=pitts train_test_ratio=0.1 pitts.sim_dim=1000
CUDA_VISIBLE_DEVICES=6 poetry run python scripts/bimodal_shuffled_data.py dataset=pitts train_test_ratio=0.1 pitts.sim_dim=2000
