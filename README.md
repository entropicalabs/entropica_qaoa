# QAOA
A module for Rigettis Forest ecosystem that implements a QAOA and VQE that is more modular than the one already packaged in Grove. An outline of what can hopefully be found here in a couple of weeks is in `PackageOutline.md`

## Documentation
Good question, read the code for now.

## Installation
Gonna need to figure out, how to turn this into a installable python package. For now just clone the repo and figure it our yourself.

## Testing
 - `pytest` runs the default tests and skips longer tests that need heavy simulations
 - `pytest --runslow` runs all tests, including the ones with heavy simulations. I didn't fix the seed for the random generator yet, so sometimes the test doesn't even work.... If it takes longer than 5 Minutes just kill the tests and restart it.
 - `pytest -s (--runslow)` is a bit more verbose and also prints all the print statements
 - with `pytest tests/<testfile>` single tests can be run to check single modules.

## Contributing
Lets be real, this is Entropica internal for now.
