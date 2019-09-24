# Changelog

## [v1.1-beta](https://gitlab.com/entropica/entropica_qaoa/tree/dev) (in development)

### Improvements and changes
- Variable names in `qaoa.cost_function.py` have been updated to better reflect
  their counterparts in `qaoa.parameters.py`. This fixed Issue #40 
  (@jlbosse, !6)
- Cost functions on the Wavefunction Simulator use
  `WavefunctionSimulator.expectation()` instead of `wf.conj()@ham@wf` which
  speeds things massively up. This fixed Issue #43.
  (@jlbosse, !7)


## [v1.0-beta](https://github.com/entropicalabs/entropica_qaoa/releases/tag/1.0)

- Initial beta release
