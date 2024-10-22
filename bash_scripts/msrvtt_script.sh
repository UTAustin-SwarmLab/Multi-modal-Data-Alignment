poetry run python mmda/any2any_conformal_retrieval.py dataset=MSRVTT
poetry run python mmda/any2any_conformal_retrieval.py dataset=MSRVTT MSRVTT.audio_encoder=clap MSRVTT.img_encoder=clip
poetry run python mmda/any2any_conformal_retrieval.py dataset=MSRVTT MSRVTT.mask_ratio=0 MSRVTT.audio_encoder=clap MSRVTT.img_encoder=clip
