# Contributing

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)

## How Can I Contribute?

### Issues

Filling issues is a great and easy way to help find bugs and get new features implemented.

#### Bugs

If you're reporting a security issue, please email me at security@zera.tax ([gpg key](https://keybase.io/zeratax/pgp_keys.asc)), otherwise
if you're reporting a bug, please create an [issue](https://github.com/ZerataX/yacx/issues/new?labels=bug&template=bug_report.md).

#### Feature Requests

If you have an idea for a new feature or an enhancement, please create an [issue](https://github.com/ZerataX/yacx/issues/new?labels=enhancement&template=feature_request.md).

### Pull Requests

Every PR should not break tests and ideally include tests for added functionality as well as documentation.

#### How to Send Pull Requests

Everyone is welcome to contribute code to yacx via GitHub pull requests (PRs).

To create a new PR, fork the project in GitHub and clone the upstream repo:

```console
$ git clone https://github.com/zeratax/yacx.git
```

Add your fork as an origin:

```console
$ git remote add fork https://github.com/YOUR_GITHUB_USERNAME/yacx.git
```

##### Build:

```console
cmake -H. -Bbuild
make -C build
```

##### Test:
```console
make -C build test
```

or

```console
pushd build/test/bin
./tests
```

Check out a new branch, make modifications and push the branch to your fork:

```console
$ git checkout -b <feature>
# edit files
$ git commit
$ git push fork <feature>
```

Open a pull request against the main yacx repo.

## Development

### C++

yacx tries to follow the [CppCoreGuidlines](http://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines) as much as possible

Code formatting will be enforced automatically via [clang-format](./.clang-format)

Tests are written using the [Catch2](https://github.com/catchorg/Catch2) test-framework

### Java

Tests are written using [Junit](https://junit.org/junit5/)
