# cosmos
CUDA_VISIBLE_DEVICES=6 poetry run python mmda/bimodal_mislabeled_data.py dataset=cosmos cosmos.sim_dim=10
CUDA_VISIBLE_DEVICES=6 poetry run python mmda/bimodal_mislabeled_data.py dataset=cosmos cosmos.sim_dim=20
CUDA_VISIBLE_DEVICES=6 poetry run python mmda/bimodal_mislabeled_data.py dataset=cosmos cosmos.sim_dim=50
CUDA_VISIBLE_DEVICES=6 poetry run python mmda/bimodal_mislabeled_data.py dataset=cosmos cosmos.sim_dim=100
CUDA_VISIBLE_DEVICES=6 poetry run python mmda/bimodal_mislabeled_data.py dataset=cosmos cosmos.sim_dim=200
CUDA_VISIBLE_DEVICES=6 poetry run python mmda/bimodal_mislabeled_data.py dataset=cosmos cosmos.sim_dim=500
CUDA_VISIBLE_DEVICES=6 poetry run python mmda/bimodal_mislabeled_data.py dataset=cosmos train_test_ratio=0.2 cosmos.sim_dim=10
CUDA_VISIBLE_DEVICES=6 poetry run python mmda/bimodal_mislabeled_data.py dataset=cosmos train_test_ratio=0.2 cosmos.sim_dim=20
CUDA_VISIBLE_DEVICES=6 poetry run python mmda/bimodal_mislabeled_data.py dataset=cosmos train_test_ratio=0.2 cosmos.sim_dim=50
CUDA_VISIBLE_DEVICES=6 poetry run python mmda/bimodal_mislabeled_data.py dataset=cosmos train_test_ratio=0.2 cosmos.sim_dim=100
CUDA_VISIBLE_DEVICES=6 poetry run python mmda/bimodal_mislabeled_data.py dataset=cosmos train_test_ratio=0.2 cosmos.sim_dim=200
CUDA_VISIBLE_DEVICES=6 poetry run python mmda/bimodal_mislabeled_data.py dataset=cosmos train_test_ratio=0.2 cosmos.sim_dim=500
