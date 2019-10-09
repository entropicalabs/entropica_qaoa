# Changelog

## [v1.2](https://gitlab.com/entropica/entropica_qaoa/tree/dev) (October 9, 2019)

### Improvements and changes
- Variable names in `qaoa.cost_function.py` have been updated to better reflect
  their counterparts in `qaoa.parameters.py`. 
- Sped up cost function calls and construction for `PrepareAndMeasureOnWFSim`
  and `QAOACostFunOnWFSim`. For very small systems and hamiltonians
  `WavefunctionSimulator().expectation()` is still faster, than our implementation. 
- Added an example of using QAOA to solve a simple QUBO problem.
- Sped up tests by removing all `with local_qvm()` context managers. You have
  to manually start a qvm in the background for the tests now!
- Added `sample_qaoa_bistrings` to `utilities.py`

### Bugfixes
- Can access `FourierExtendedParams.z_rotation_angles` now for hamiltonians
  without bias terms. 

## [v1.0-beta](https://github.com/entropicalabs/entropica_qaoa/releases/tag/1.0) (September 12, 2019)

- Initial beta release
