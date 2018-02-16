.venv: .venv/bin/activate
.venv/bin/activate: requirements.txt
	test -d .venv || virtualenv -p python3 .venv
	.venv/bin/pip install -Ur requirements.txt
