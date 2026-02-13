.PHONY: all version localdocs publicdocs clean

# version format: major.minor.patch; to bump patch, do "make version v=patch"
version:
	@current=$$(grep -Po '(?<=^version = ")[^"]+' pyproject.toml); \
	IFS='.' read -r major minor patch <<< "$$current"; \
	case "$(v)" in \
		major) major=$$((major + 1)); minor=0; patch=0 ;; \
		minor) minor=$$((minor + 1)); patch=0 ;; \
		patch) patch=$$((patch + 1)) ;; \
		*) echo "Usage: make version v=major|minor|patch"; exit 1 ;; \
	esac; \
	new="$$major.$$minor.$$patch"; \
	sed -i "s/^version = \".*\"/version = \"$$new\"/" pyproject.toml; \
	git add pyproject.toml; \
	git commit -m "v$$new"; \
	git tag "v$$new"; \
	git push; \
	git push --tags; \
	echo "Bumped version to $$new"

localdocs:
	@uv run mkdocs build

servedocs: export JUPYTER_EXECUTE=false
servedocs:
	@uv run mkdocs serve


publicdocs:
	@uv run mkdocs gh-deploy
