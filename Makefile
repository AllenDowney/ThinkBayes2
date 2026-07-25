PROJECT_NAME = ThinkBayes2
PYTHON_VERSION = 3.12

.PHONY: help create_environment create_environment_dev delete_environment \
	update_environment update_environment_dev clean tests

help:
	@echo "Available targets:"
	@echo "  create_environment     - Create conda env from environment.yml"
	@echo "  create_environment_dev - Base env + environment-dev.yml extras"
	@echo "  delete_environment     - Remove conda environment"
	@echo "  update_environment     - Update base env (with --prune)"
	@echo "  update_environment_dev - Update base + dev extras"
	@echo "  clean                  - Remove temporary files and caches"
	@echo "  tests                  - Run nbmake on soln chap01–18"

create_environment:
	mamba env create -y -f environment.yml
	@echo ">>> Environment created. Activate with:\nconda activate $(PROJECT_NAME)"

create_environment_dev: create_environment
	mamba env update -y -f environment-dev.yml --name $(PROJECT_NAME)
	@echo ">>> Dev environment ready. Activate with:\nconda activate $(PROJECT_NAME)"

delete_environment:
	mamba env remove -y --name $(PROJECT_NAME)
	@echo ">>> Environment $(PROJECT_NAME) removed"

update_environment:
	mamba env update -y -f environment.yml --name $(PROJECT_NAME) --prune
	@echo ">>> Environment updated"

update_environment_dev: update_environment
	mamba env update -y -f environment-dev.yml --name $(PROJECT_NAME)
	@echo ">>> Dev environment updated"

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

tests:
	cd soln; pytest --nbmake chap0[1-9].ipynb
	cd soln; pytest --nbmake chap1[0-8].ipynb
