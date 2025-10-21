# Repository Guidelines

## Project Structure & Module Organization
The SDK lives in `relai/`, with subpackages such as `critico`, `maestro`, and `mocker`, plus shared clients in `_client.py`, `simulator.py`, and `schema/`. Tests sit in `tests/` mirroring the package layout; add or extend fixtures in `tests/conftest.py` rather than scattering helpers. Reference agents and scripts belong in `examples/`. Documentation sources are under `docs/` with theme overrides in `docs_overrides/`; `site/` holds the rendered MkDocs build and `dist/` captures published wheels.

## Build, Test, and Development Commands
- `uv sync --group dev` installs runtime and development dependencies; rerun after editing `pyproject.toml` or `uv.lock`.
- `uv run pytest` executes the async-enabled suite; target a module with `uv run pytest tests/relai/test_client.py`.
- `uv run pytest --cov=relai --cov-report=term-missing` checks coverage; keep touched modules at or above their prior baseline.
- `uv run python -m build` produces distribution artifacts in `dist/`; clear `build/` if metadata changes before publishing.

## Coding Style & Naming Conventions
Write Python 3.10+ code with four-space indentation and full type hints on public APIs. Modules stay lowercase_with_underscores, classes use `PascalCase`, and async helpers should read as verbs (`get_user_query`). Keep user-facing constants near their module entry points and export shared models from package `__init__.py` files for discoverability. Run `ruff` before committing; it autocorrects import order and blocks obvious runtime pitfalls.

## Testing Guidelines
Pytest with `pytest-asyncio` backs all suites. Mark scope explicitly using `@pytest.mark.unit` or `@pytest.mark.integration` to keep filters meaningful. Favor fixtures in `tests/conftest.py` for shared clients, personas, and mock transport layers. For new evaluators or simulators, add async success and failure cases and confirm coverage with `pytest --cov`.

## Commit & Pull Request Guidelines
Match the existing history with concise, imperative commit subjects (`Fix link`, `Add tutorial`) under 72 characters. Reference related issues or docs in the body. PRs should outline intent, affected subpackages, and validation steps (`uv run pytest`, `uv run ruff format`, `uv run ruff check --fix`). Attach screenshots or trace snippets when altering user-visible behavior, and wait for green CI before requesting merge.

## Security & Configuration Tips
Store credentials such as `RELAI_API_KEY`, `OPENAI_API_KEY`, and telemetry tokens in your shell environment. Avoid committing `.env` contents—update `.gitignore` for any new secret-bearing paths. Scrub simulation tapes and persona definitions before sharing examples outside `examples/`, and keep production-only endpoints out of default configs.
