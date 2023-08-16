# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]

### Added
* FLAMINGO support

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
