# Claude Development Notes for scida

## Project Overview

**scida** is a Python library for scalable analysis of large scientific datasets, primarily targeting astrophysics simulations (particle-based cosmological simulations) and observational data. It provides a unified interface to load, analyze, and process big data using **dask** for parallel/lazy computation.

**Key Features:**
- Unified high-level interface via `scida.load()`
- Parallel, task-based data processing with dask arrays
- Physical unit support via pint
- Extensible architecture with mixins and custom dataset classes
- Support for HDF5, zarr, and FITS file formats

## Project Setup

This project uses **uv** for dependency management. All Python commands should be prefixed with `uv run`.

### Installation
```bash
uv sync                    # Install all dependencies
uv sync --all-groups       # Include dev dependencies
```

### Running Commands
```bash
uv run python -m pytest tests/test_init.py -v   # Run specific test
uv run python -c "import scida; print('test')"  # Run Python code
uv run mypy src/                                 # Type checking
uv run black src/                                # Code formatting
uv run isort src/                                # Import sorting
```

### Common Development Commands
| Task | Command |
|------|---------|
| Run all tests | `uv run pytest` |
| Run tests with coverage | `uv run coverage run -m pytest` |
| Run specific test file | `uv run pytest tests/test_interface.py -v` |
| Build docs locally | `make servedocs` (serves at localhost:8000) |
| Deploy docs | `make publicdocs` |
| Version bump | `make version v=patch` (or minor/major) |

### Git Worktree Workflow for Feature Development

**Always develop features on separate branches using git worktrees.** This keeps the main working directory clean and allows parallel work on multiple features.

```bash
# Create a new feature branch and worktree
git branch feature/my-feature main
git worktree add branches/my-feature feature/my-feature

# Work in the worktree
cd branches/my-feature
# ... make changes, commit, etc.

# When done, clean up
cd /home/cbyrohl/repos/scida
git worktree remove branches/my-feature
git branch -d feature/my-feature  # after merging
```

**Directory structure:**
```
/home/cbyrohl/repos/scida/
├── branches/                    # All worktrees live here
│   ├── fix-validate-path-types/ # Example worktree
│   └── feature-xyz/             # Another worktree
├── src/                         # Main repo (stay on main)
└── ...
```

**Key rules:**
- Never commit feature work directly to `main` in the main repo
- Each refactoring task in `refactoring_tasks/` specifies its branch name
- Run tests in the worktree before creating PRs
- The `branches/` directory is gitignored

## Architecture Overview

### Directory Structure
```
src/scida/
├── __init__.py          # Package entry point, imports key classes
├── convenience.py       # load() function - main user entry point
├── interface.py         # BaseDataset, Dataset, Selector base classes
├── fields.py            # FieldContainer - data storage abstraction
├── series.py            # DatasetSeries - collection of datasets
├── registries.py        # Type registries for auto-detection
├── discovertypes.py     # Type detection logic
├── config.py            # Configuration management
├── io/                  # I/O backends (HDF5, zarr, FITS)
│   ├── _base.py         # ChunkedHDF5Loader, load functions
│   └── fits.py          # FITS file support
├── interfaces/mixins/   # Composable functionality
│   ├── units.py         # UnitMixin - pint unit support
│   ├── cosmology.py     # CosmologyMixin - cosmological parameters
│   └── spatial.py       # SpatialCartesian3DMixin
├── customs/             # Simulation-specific implementations
│   ├── gadgetstyle/     # Base Gadget-style simulations
│   ├── arepo/           # Arepo/Illustris/TNG simulations
│   ├── swift/           # SWIFT simulations
│   ├── gizmo/           # GIZMO/FIRE simulations
│   └── rockstar/        # Rockstar halo catalogs
└── configfiles/         # YAML config files (units, simulations)
```

### Core Classes

#### `BaseDataset` / `Dataset` (interface.py)
- Base class for all datasets
- Uses `MixinMeta` metaclass for dynamic mixin composition
- Auto-registers subclasses in `dataset_type_registry`
- Key methods: `__init__()`, `validate_path()`, `save()`, `info()`

#### `FieldContainer` (fields.py)
- MutableMapping that stores dask arrays and supports lazy field evaluation
- Supports nested containers (e.g., `data["PartType0"]["Coordinates"]`)
- Field recipes: functions that compute derived fields on-demand
- Key methods: `register_field()`, `keys()`, `get_dataframe()`

#### `DatasetSeries` (series.py)
- Container for multiple related datasets (e.g., simulation snapshots)
- Lazy initialization via `delay_init` decorator
- Metadata caching for efficient series navigation

#### Mixins (interfaces/mixins/)
- **UnitMixin**: Adds pint unit support, reads unit config files
- **CosmologyMixin**: Cosmological parameter handling (redshift, scale factor)
- **SpatialCartesian3DMixin**: Spatial operations in Cartesian coordinates

### Type Detection Flow

1. `scida.load(path)` is called
2. `_determine_type()` iterates through registered types
3. Each type's `validate_path()` classmethod is called
4. Most specific matching type is selected (via MRO analysis)
5. `_determine_mixins()` adds appropriate mixins
6. Dataset instance is created with dynamic class composition

### Registry Pattern
```python
# In registries.py
dataset_type_registry: Dict[str, Type] = {}
dataseries_type_registry: Dict[str, Type] = {}
mixin_type_registry: Dict[str, Type] = {}

# Auto-registration via __init_subclass__
class MyDataset(Dataset):  # Automatically registered
    @classmethod
    def validate_path(cls, path, **kwargs):
        # Return CandidateStatus.MAYBE or CandidateStatus.NO
        ...
```

## Creating Custom Dataset Classes

### Pattern for New Dataset Types
```python
from scida.discovertypes import CandidateStatus
from scida.interface import Dataset

class MyCustomDataset(Dataset):
    """Custom dataset for specific simulation type."""

    @classmethod
    def validate_path(cls, path, *args, **kwargs) -> CandidateStatus:
        """Check if path matches this dataset type."""
        # Check file structure, metadata, etc.
        if is_valid:
            return CandidateStatus.MAYBE
        return CandidateStatus.NO

    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)
        # Custom initialization
        # Access metadata via self._metadata_raw
        # Access data via self.data (FieldContainer)
```

### Registering Derived Fields
```python
@snap.data.register_field("PartType0", name="Temperature")
def temperature(container, snap=None, ureg=None):
    """Compute temperature from internal energy."""
    u = container["InternalEnergy"]
    # ... computation ...
    return temperature_array
```

## Testing

### Test Structure
```
tests/
├── conftest.py              # Fixtures, test configuration
├── helpers.py               # Dummy file generators (DummyGadgetFile, etc.)
├── testdata_properties.py   # Test data requirement decorators
├── test_interface.py        # Core interface tests
├── test_units.py            # Unit handling tests
├── customs/                 # Simulation-specific tests
│   ├── test_arepo.py
│   ├── test_swift.py
│   └── ...
└── simulations/             # Integration tests with real data
```

### Test Data Handling
- Tests use `@require_testdata` and `@require_testdata_path` decorators
- Test data path set via `SCIDA_TESTDATA_PATH` environment variable
- Dummy files created via `DummyGadgetSnapshotFile`, `DummyTNGFile` classes
- Cache directory isolated per test via `cachedir` fixture

### Running Tests
```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_interface.py

# Run tests matching pattern
uv run pytest -k "arepo"

# Run with coverage
uv run pytest --cov=scida

# Run via nox (multi-Python testing)
nox -s tests
```

### Nox Sessions
- `tests`: Run tests across Python 3.9-3.12
- `tests_numpy_versions`: Test with different NumPy versions
- `coverage`: Generate coverage report

## Configuration

### User Configuration
- Location: `~/.config/scida/config.yaml`
- Copied from `scida/configfiles/config.yaml` on first use
- Override with `SCIDA_CONFIG_PATH` environment variable

### Environment Variables
| Variable | Purpose |
|----------|---------|
| `SCIDA_CONFIG_PATH` | Override config file location |
| `SCIDA_CACHE_PATH` | Override cache directory |
| `SCIDA_TESTDATA_PATH` | Test data directory |

### Unit Configuration
- General units: `scida/configfiles/units/general.yaml`
- Simulation-specific: Pass `unitfile` parameter to `load()`
- Unit definitions use pint format

## Documentation

### Building Docs
```bash
make servedocs     # Serve locally with hot reload
make localdocs     # Build static site
make publicdocs    # Deploy to GitHub Pages
```

### Documentation Stack
- **MkDocs** with Material theme
- **mkdocstrings** for API docs (NumPy docstring style)
- **mkdocs-jupyter** for notebook integration

## Code Conventions

### Style Guidelines
- **Formatter**: black
- **Import sorting**: isort
- **Linting**: flake8
- **Type hints**: Used throughout, check with mypy

### Docstring Format (NumPy style)
```python
def function(param1: str, param2: int = 0) -> bool:
    """
    Short description.

    Parameters
    ----------
    param1 : str
        Description of param1.
    param2 : int, optional
        Description of param2.

    Returns
    -------
    bool
        Description of return value.
    """
```

### Logging
```python
import logging
log = logging.getLogger(__name__)
log.debug("Debug message")
log.info("Info message")
```

## Key Patterns

### Lazy Evaluation
- Data loaded as dask arrays (lazy by default)
- Call `.compute()` to materialize results
- Field recipes evaluated on first access

### Virtual Caching
- HDF5 datasets loaded as virtual datasets when possible
- Reduces memory footprint for multi-file datasets
- Controlled via `virtualcache` parameter

### Mixin Composition
```python
# Mixins dynamically composed at load time
instance = cls(path, mixins=[UnitMixin, CosmologyMixin], **kwargs)

# Results in dynamically created class:
# DatasetWithUnitMixinAndCosmologyMixin
```

### CandidateStatus for Type Detection
```python
from scida.discovertypes import CandidateStatus

class MyDataset(Dataset):
    @classmethod
    def validate_path(cls, path, **kwargs):
        if definitely_this_type:
            return CandidateStatus.YES
        if possibly_this_type:
            return CandidateStatus.MAYBE
        return CandidateStatus.NO
```

## Common Gotchas

1. **Always use `uv run`** for Python commands
2. **Dask arrays are lazy** - use `.compute()` to get values
3. **Units via pint** - access magnitude with `.magnitude`
4. **Test data required** - many tests need `SCIDA_TESTDATA_PATH` set
5. **Metadata in `_metadata_raw`** - raw HDF5 attributes stored here
6. **Field recipes vs fields** - recipes are functions, fields are arrays

## Useful Development Patterns

### Inspecting Dataset Structure
```python
ds = scida.load("path/to/data")
ds.info()  # Print structure overview
ds.data.keys()  # List top-level containers
ds.data["PartType0"].keys()  # List fields
```

### Debugging Type Detection
```python
import logging
logging.basicConfig(level=logging.DEBUG)
ds = scida.load("path/to/data")
# Check logs for type detection info
```

### Working with Units
```python
ds = scida.load("path", units="cgs")  # Load with CGS units
arr = ds.data["PartType0"]["Coordinates"]
arr.units  # Check units
arr.magnitude  # Get raw values
arr.to("kpc")  # Convert units
```
