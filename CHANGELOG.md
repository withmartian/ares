# Changelog

All notable changes to ARES will be documented in this file.

ARES follows semantic versioning, but `0.y.z` releases are initial development and may include breaking changes.

## [0.1.0] - 2026-06-05

### Added

- Added the ARES proxy for HTTP-mediated LLM request interception.
- Added SkyRL integration example.
- Added a transformers-backed local LLM client in `ares.contrib`.
- Added mechanistic interpretability tooling and a 20Q case study.
- Added explicit Daytona configuration support for container factories.

### Changed

- Versioned duplicate Harbor-derived registry preset IDs so task names disambiguate dataset versions.
- Refactored LLM request conversion into dedicated converter modules.
- Switched SWE-bench runs to the repo-owned mini-swe-agent configuration.
