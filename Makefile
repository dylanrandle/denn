run_docker:
	docker run -it --rm dylanrandle/denn bash

export_env:
	conda env export | grep -v "prefix" > environment.yml
	# --no-builds for removing build ids
	conda env export --no-builds | grep -v "prefix" > environment_no_builds.yml
	# this just captures what I explicitly typed in for conda (not pip)
	# must include pip packages manually...
	conda env export --from-history | grep -v "prefix" > environment_from_history.yml
