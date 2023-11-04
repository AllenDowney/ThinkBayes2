PROJECT_NAME = ThinkBayes2
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python


create_environment:
	conda create -y --name $(PROJECT_NAME) python=$(PYTHON_VERSION) pymc
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"


requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt


## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


tests:
	cd soln; pytest --nbmake chap0[1-9].ipynb
	cd soln; pytest --nbmake chap1[0-8].ipynb
