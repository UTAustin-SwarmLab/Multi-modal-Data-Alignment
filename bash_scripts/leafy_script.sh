# CUDA_VISIBLE_DEVICES=1 poetry run python mmda/bimodal_classification.py dataset=leafy_spurge leafy_spurge.sim_dim=10
# CUDA_VISIBLE_DEVICES=1 poetry run python mmda/bimodal_classification.py dataset=leafy_spurge leafy_spurge.sim_dim=50
# CUDA_VISIBLE_DEVICES=1 poetry run python mmda/bimodal_classification.py dataset=leafy_spurge leafy_spurge.sim_dim=100
# CUDA_VISIBLE_DEVICES=1 poetry run python mmda/bimodal_classification.py dataset=leafy_spurge leafy_spurge.sim_dim=250

CUDA_VISIBLE_DEVICES=1 poetry run python mmda/linear_svm_clip.py dataset=leafy_spurge leafy_spurge.sim_dim=250 train_test_ratio=0.4
CUDA_VISIBLE_DEVICES=1 poetry run python mmda/linear_svm_clip.py dataset=leafy_spurge leafy_spurge.sim_dim=250 train_test_ratio=0.6
CUDA_VISIBLE_DEVICES=1 poetry run python mmda/linear_svm_clip.py dataset=leafy_spurge leafy_spurge.sim_dim=250 train_test_ratio=0.7
CUDA_VISIBLE_DEVICES=1 poetry run python mmda/linear_svm_clip.py dataset=leafy_spurge leafy_spurge.sim_dim=250 train_test_ratio=0.888
