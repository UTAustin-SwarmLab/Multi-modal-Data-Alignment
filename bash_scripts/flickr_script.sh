# flickr
# poetry run python mmda/bimodal_retrieval.py dataset=flickr flickr.sim_dim=750 flickr.equal_weights=True flickr.img2text=True
# poetry run python mmda/bimodal_retrieval.py dataset=flickr flickr.sim_dim=750 flickr.equal_weights=True flickr.img2text=False
# poetry run python mmda/bimodal_retrieval.py dataset=flickr flickr.sim_dim=750 flickr.img2text=True
# poetry run python mmda/bimodal_retrieval.py dataset=flickr flickr.sim_dim=750 flickr.img2text=False
poetry run python mmda/retrieval_spearman_coeff.py dataset=flickr flickr.equal_weights=True flickr.img2text=True
poetry run python mmda/retrieval_spearman_coeff.py dataset=flickr flickr.equal_weights=True flickr.img2text=False
poetry run python mmda/retrieval_spearman_coeff.py dataset=flickr flickr.img2text=True
poetry run python mmda/retrieval_spearman_coeff.py dataset=flickr flickr.img2text=False