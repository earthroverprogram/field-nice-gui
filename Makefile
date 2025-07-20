.DEFAULT_GOAL:= help  # because it's is a safe task.

.PHONY: help
help: # Show help for each of the makefile recipes.
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done

run: # Run the application.
	uv run app.py

venv:  # Create an empty virtual environment (enough to create the requirements files).
	uv venv .venv
	uv sync
