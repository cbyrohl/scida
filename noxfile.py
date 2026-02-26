from pathlib import Path

import nox

# Use uv as the default venv backend
nox.options.default_venv_backend = "uv"

python_versions = ["3.13", "3.12", "3.11"]
python_dflt = "3.11"
numpy_versions = ["1.26", "2.0"]


def tests_base(session: nox.Session) -> None:
    session.install("coverage[toml]", "pytest", "pytest-mock", "pygments")
    session.install(".")
    try:
        session.run(
            "coverage",
            "run",
            "--parallel-mode",
            "-m",
            "pytest",
            "-v",
            "--junitxml=report.xml",
            *session.posargs,
        )
    finally:
        if session.interactive:
            session.notify("coverage", posargs=[])


@nox.session(python=python_versions)
def tests(session: nox.Session) -> None:
    tests_base(session)


@nox.session(python=python_versions)
@nox.parametrize("numpy", numpy_versions)
def tests_numpy_versions(session: nox.Session, numpy: str) -> None:
    session.install(f"numpy=={numpy}")
    tests_base(session)


@nox.session(python=python_dflt)
def coverage(session: nox.Session) -> None:
    args = session.posargs or ["report"]
    session.install("coverage[toml]")
    if not session.posargs and any(Path().glob(".coverage.*")):
        session.run("coverage", "combine")
    session.run("coverage", *args)
