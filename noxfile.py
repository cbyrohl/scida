from pathlib import Path

from nox_poetry import Session, session

python_versions = ["3.10", "3.9"]
python_dflt = "3.10"


@session(python=python_versions)
def tests(session):
    session.install("coverage[toml]", "pytest", "pygments")
    session.install(".")
    try:
        session.run(
            "coverage",
            "run",
            "--parallel",
            "-m",
            "pytest",
            "-v",
            "--junitxml=report.xml",
            *session.posargs
        )
    finally:
        if session.interactive:
            session.notify("coverage", posargs=[])


@session(python=python_dflt)
def coverage(session: Session) -> None:
    args = session.posargs or ["report"]
    session.install("coverage[toml]")
    if not session.posargs and any(Path().glob(".coverage.*")):
        session.run("coverage", "combine")
    session.run("coverage", *args)
