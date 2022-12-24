from nox_poetry import session


@session(python=["3.10", "3.9"])
def tests(session):
    session.install("pytest", ".")
    session.run("pytest")
