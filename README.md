# QAOA
A module for Rigettis Forest ecosystem that implements a QAOA and VQE that is more modular than the one already packaged in Grove.

An outline of what can hopefully be found here in a couple of weeks is in `PackageOutline.md`

## Documentation
Good question, read the code for now.

## Installation
Installation of this package can be performed in a few simple steps.
1. Open terminal and enter the site-packages folder of your preferred Python environment.

For those with Anaconda installed, the command looks like:
```
cd /anaconda3/envs/<my-env>/lib/pythonX.Y/site-packages/
```
For those unsure of the location of their site-packages folder, you can simply run 'pip show <package name>' and your terminal will display the directory location of your python packages.

2. Clone the forest_qaoa repository into your site-packages folder.
```
git clone <put http here>
```
3. Enter the forest_qaoa folder and run the following command
 
```
python setup.py install
```
You can now import this package as you would any conda- or pip-installed library!

## Testing
 - `pytest` runs the default tests and skips longer tests that need heavy simulations
 - `pytest --runslow` runs all tests, including the ones with heavy simulations. I didn't fix the seed for the random generator yet, so sometimes the test doesn't even work.... If it takes longer than 5 Minutes just kill the tests and restart it.
 - `pytest -s (--runslow)` is a bit more verbose and also prints all the print statements
 - with `pytest tests/<testfile>` single tests can be run to check single modules.

## Contributing
Lets be real, this is Entropica internal for now.
