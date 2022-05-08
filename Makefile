
all: prebuild run

prebuild:
	@pip install -r requirements.txt

run:
	@python src/main.py