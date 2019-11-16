# Development Routines of the EntropicaQAOA package

<img src="branches.png" alt="Branch model of the `entropica_qaoa` repository" style="zoom:20%;" />

​		The branches of the `entropica_qaoa` repository

## The different branches

- **`master`**: This branch always contains the last published version of the package and mirrors exactly what is in `pip` currently. It is also identical to the `master` branch on github. Each new commit here gets a tag with ("V1.0" and "V1.1") the updated version number and also a new version tag on github.
- **`dev`**: The main development branch. It always contains the latest _stable_ version of the codebase. This means, that before pushing to it from your local machine you make sure that all tests run and that all new features are complete (i.e. no half finished functions, sentences etc. here. That is what `feature` branches are for)
- **`relase_#`**: Each new major release (i.e. the first digit of the version number changes) gets its own release branch ("release v1.0").  If there are hotfixes that _need_ to be fixed within this version (i.e. can not wait for the next major version) they are fixed in this branch and then merged into `master` ("merge hotfix into master") _and_ `dev` ("merge hotfix into dev"). It is crucial, that these changes are also merged back into `dev`. Only then we can guarantee, that the next time we create a new major release and merge from `dev` to `master` (via a new `release_#+1` branch) we get no merge conflicts.
- **`feature`**: New features are always developed in their own branches ("Started work on new feature"). If there are important changes on `dev` during development of the new feature ("merge other feature into dev") they can also be merged into the feature branch ("update from dev"). Once the feature is done ("developed new feature") and tested it is merged back into dev ("merge feature into dev").