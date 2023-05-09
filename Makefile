.PHONY: all version localdocs publicdocs clean

version:
	@poetry version $(v)
	@git add pyproject.toml
	@git commit -m "v$$(poetry version -s)"
	@git tag v$$(poetry version -s)
	@git push
	@git push --tags
	@poetry version

localdocs:
	@poetry run mkdocs build

servedocs: export JUPYTER_EXECUTE=false
servedocs:
	@poetry run mkdocs serve

publicdocs:
	@poetry run mkdocs build
	@rsync -arv site/* hetzner:/home/cbyrohl/public_content/astrodask
