PYTHON ?= python3

.PHONY: install train-cv train-rl run

install:
	$(PYTHON) -m pip install -r cv_pipeline/requirements.txt

train-cv:
	$(PYTHON) -m cv_pipeline.detection.train --config cv_pipeline/config.yaml

train-rl:
	$(PYTHON) RL_Agent/train_v7_mc.py

run:
	@if [ -z "$(SOURCE)" ]; then echo "Usage: make run SOURCE=path/to/video.mp4"; exit 1; fi
	$(PYTHON) -m cv_pipeline.pipeline.session --source $(SOURCE) --config cv_pipeline/config.yaml
