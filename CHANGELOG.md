# Changelog

## [v1.1-beta](https://gitlab.com/entropica/entropica_qaoa/tree/dev) (in development)

### Improvements and changes
- Variable names in `qaoa.cost_function.py` have been updated to better reflect
  their counterparts in `qaoa.parameters.py`. This fixed Issue #40 
  (@jlbosse, !6)
- Sped up cost function calls and construction for `PrepareAndMeasureOnWFSim`
  and `QAOACostFunOnWFSim`. For very small systems and hamiltonians
  `WavefunctionSimulator().expectation()` is still faster, than our implementation. This fixed Issue #43 and #44.
  (@jlbosse, !7 and d6df9f73e504c92b06485e1ce8e95676f7a02ccd)
- Added an example of using QAOA to solve a simple QUBO problem.


## [v1.0-beta](https://github.com/entropicalabs/entropica_qaoa/releases/tag/1.0)

- Initial beta release
