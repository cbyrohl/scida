# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]

### Changed

- unit config files are searched relative to working directory

## [0.3.2] - 2024-05-23

### Added

- for arepo snapshots, add local subhalo id selector (PR #163)
- JOSS publication reference in readme (PR #159)
- add new question to FAQ docs
- move docs to scida.io

### Fixed

- bugfix for potentially wrong snapshot order for gadgetsyle series
- gh pipeline fix required due to new abs path requirement in some pip calls (PR #161)
- update outdated config path in config (PR #162)
- fix wget command in docs (#156)
- some fixes to unit discovery
- fix on parallel cache file access

## [0.3.1] - 2024-02-19

- Cut a new release for zenodo record as required for JOSS publication

## [0.3.0] - 2024-02-16

### Added

- various documentation improvements, including API cleanup, cookbooks and developer guide (PR #123, #136, #137, #145, #153; furthermore fixes #147, #148, #149, #151)
- improved docstring coverage (PR #121)
- more default units for gadget-style snapshots (PR #96)
- unit inference reporting (PR #130)
- tolerance for get_dataset (PR #132)
- overwrite_cache for load of series (PR #152)

### Fixed

- recover from corrupt cache (PR #95)
- various documentation fixes (PR #120, #107, #112, #114, #122, #124, #126, #127)
- testing of markdown docs (PR #128, #135, #146)
- some unit support fix (PR #131)


## [0.2.4] - 2023-08-24

### Added

- FLAMINGO support
- proof-of-concept support for FITS format
- configuration documentation
- py3.11 support
- unit support for field recipes

### Fixed

- MTNG catalog detection
- various minor bugs


## [0.2.3] - 2023-08-14

### Added
- test docs
- simple info() for series
- support unbound=True/subhaloID=... operator for return-data of Arepo snapshots
- support nmax=.../idxlist=... for map-group-operation/grouped
- support per-subhalo operations in grouped
- introduce "unknown" units
- add "LocalSubhaloID" and "SubhaloID" field for Arepo snapshot particles
- Lgalaxies support
- optional sanity check for dataset length in GadgetStyleSnapshot

### Fixed
- various bugs related to dask operations using pint quantities.
- detection for TNG100-2/3
- unit/shape detection for custom function for grouped operations
- fix incorrect output hape of SubhaloIndex for black holes
- improvements/fixes to docs

## [0.2.2] - 2023-07-11

### Added

- Add basic MTNG support
- Add basic TNG-Cluster support
- Add basic FIRE2 support

## [0.2.1] - 2023-06-29

- Still no changelog. Some bug fixes.

## [0.2.0] - 2023-06-21

- First PyPI release
