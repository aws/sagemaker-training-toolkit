# Contributing Guidelines

Thank you for your interest in contributing to our project. Whether it's a bug report, new feature, correction, or additional
documentation, we greatly value feedback and contributions from our community.

Please read through this document before submitting any issues or pull requests to ensure we have all the necessary
information to effectively respond to your bug report or contribution.


## Submitting bug reports and feature requests

We welcome you to use the GitHub issue tracker to report bugs or suggest features.

When filing an issue, please check [existing open](https://github.com/aws/sagemaker-training-toolkit/issues), or [recently closed](https://github.com/aws/sagemaker-training-toolkit/issues?utf8=%E2%9C%93&q=is%3Aissue%20is%3Aclosed%20), issues to make sure somebody else hasn't already
reported the issue. To create a new issue, select the template that most closely matches what you're writing about (ie. "Bug report", "Documentation request", or "Feature request"). Please fill out all information requested in the issue template.

## Contributing via pull requests
Contributions via pull requests are much appreciated. Before sending us a pull request, please ensure that:

- You are working against the latest source on the *master* branch.
- You check existing open, and recently merged, pull requests to make sure someone else hasn't addressed the problem already.
- You open an issue to discuss any significant work - we would hate for your time to be wasted.

To send us a pull request, please:

1. Fork the repository.
2. Modify the source; please focus on the specific change you are contributing. If you also reformat all the code, it will be hard for us to focus on your change.
3. Ensure local tests pass.
4. Commit to your fork using [clear commit messages](#committing-your-change).
5. Send us a pull request, answering any default questions in the pull request interface.
6. Pay attention to any automated CI failures reported in the pull request, and stay involved in the conversation.
   The [sagemaker-bot](https://github.com/sagemaker-bot) will comment on the pull request with a link to the build logs.

GitHub provides additional document on [forking a repository](https://help.github.com/articles/fork-a-repo/) and
[creating a pull request](https://help.github.com/articles/creating-a-pull-request/).

### Running the unit tests

1. Install tox using `pip install tox`
1. Install coverage using `pip install .[test]`
1. cd into the sagemaker-training-toolkit folder: `cd sagemaker-training-toolkit`
1. Run the following tox command and verify that all code checks and unit tests pass: `tox test/unit`

You can also run a single test with the following command: `tox -e py36 -- -s -vv test/unit/test_entry_point.py::test_install_module`  
  * Note that the coverage test will fail if you only run a single test, so make sure to surround the command with `export IGNORE_COVERAGE=-` and `unset IGNORE_COVERAGE`
  * Example: `export IGNORE_COVERAGE=- ; tox -e py36 -- -s -vv test/unit/test_entry_point.py::test_install_module ; unset IGNORE_COVERAGE`


### Running the integration tests

Our CI system runs integration tests (the ones in the `test/integration` directory), in parallel, for every pull request.  
You should only worry about manually running any new integration tests that you write, or integration tests that test an area of code that you've modified.  

1. Follow the instructions at [Set Up the AWS Command Line Interface (AWS CLI)](https://docs.aws.amazon.com/polly/latest/dg/setup-aws-cli.html).
1. To run a test, specify the test file and method you want to run per the following command: `tox -e py36 -- -s -vv test/integration/local/test_dummy.py::test_install_requirements`
   * Note that the coverage test will fail if you only run a single test, so make sure to surround the command with `export IGNORE_COVERAGE=-` and `unset IGNORE_COVERAGE`
   * Example: `export IGNORE_COVERAGE=- ; tox -e py36 -- -s -vv test/integration/local/test_dummy.py::test_install_requirements ; unset IGNORE_COVERAGE`


### Making and testing your change

1. Create a new git branch:
     ```shell
     git checkout -b my-fix-branch master
     ```
1. Make your changes, **including unit tests** and, if appropriate, integration tests.
   1. Include unit tests when you contribute new features or make bug fixes, as they help to:
      1. Prove that your code works correctly.
      1. Guard against future breaking changes to lower the maintenance cost.
   1. Please focus on the specific change you are contributing. If you also reformat all the code, it will be hard for us to focus on your change.
1. Run all the unit tests as per [Running the unit tests](#running-the-unit-tests), and verify that all checks and tests pass.
   1. Note that this also runs tools that may be necessary for the automated build to pass (ex: code reformatting by 'black').  


### Committing your change

We use commit messages to update the project version number and generate changelog entries, so it's important for them to follow the right format. Valid commit messages include a prefix, separated from the rest of the message by a colon and a space. Here are a few examples:

```
feature: support VPC config for hyperparameter tuning
fix: fix flake8 errors
documentation: add MXNet documentation
```

Valid prefixes are listed in the table below.

| Prefix          | Use for...                                                                                     |
|----------------:|:-----------------------------------------------------------------------------------------------|
| `breaking`      | Incompatible API changes.                                                                      |
| `deprecation`   | Deprecating an existing API or feature, or removing something that was previously deprecated.  |
| `feature`       | Adding a new feature.                                                                          |
| `fix`           | Bug fixes.                                                                                     |
| `change`        | Any other code change.                                                                         |
| `documentation` | Documentation changes.                                                                         |

Some of the prefixes allow abbreviation ; `break`, `feat`, `depr`, and `doc` are all valid. If you omit a prefix, the commit will be treated as a `change`.

For the rest of the message, use imperative style and keep things concise but informative. See [How to Write a Git Commit Message](https://chris.beams.io/posts/git-commit/) for guidance.


### Sending a pull request

GitHub provides additional document on [creating a pull request](https://help.github.com/articles/creating-a-pull-request/).

Please remember to:
* Use commit messages (and PR titles) that follow the guidelines under [Committing your change](#committing-your-change).
* Send us a pull request, answering any default questions in the pull request interface.
* Pay attention to any automated CI failures reported in the pull request, and stay involved in the conversation.

## Finding contributions to work on
Looking at the [existing issues](https://github.com/aws/sagemaker-training-toolkit/issues) is a great place to start.


## Code of Conduct
This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct).
For more information see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) or contact
opensource-codeofconduct@amazon.com with any additional questions or comments.


## Security issue notifications
If you discover a potential security issue in this project we ask that you notify AWS/Amazon Security via our [vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public github issue.


## Licensing

See the [LICENSE](https://github.com/aws/sagemaker-training-toolkit/blob/master/LICENSE) file for our project's licensing. We will ask you to confirm the licensing of your contribution.

We may ask you to sign a [Contributor License Agreement (CLA)](http://en.wikipedia.org/wiki/Contributor_License_Agreement) for larger changes.
