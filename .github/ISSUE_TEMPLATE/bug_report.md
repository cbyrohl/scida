---
name: Bug report
about: Create a report to help us improve
title: ''
labels: ''
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
It helps us to diagnose the problem if you can describe the steps to reproduce the behavior. 

**Expected behavior and actual outcome**
A clear and concise description of what you expected to happen, what outcome you got instead and how this differs from the expectation.

**Environment:**
Add the environment by running
```
import platform
from importlib.metadata import version
print("python version:", platform.python_version())
print("scida version:", version("scida"))
print("pint version:", version("pint"))
print("dask version:", version("dask"))
```
Please also add information about your operation system.

**Additional context**
Add any other context about the problem here.
