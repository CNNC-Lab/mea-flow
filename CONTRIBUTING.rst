Contributing to MEA-Flow
========================

We welcome contributions to MEA-Flow! This guide will help you get started.

Development Setup
-----------------

1. **Fork and Clone**

   .. code-block:: bash

      git clone https://github.com/your-username/mea-flow.git
      cd mea-flow

2. **Set up Development Environment**

   .. code-block:: bash

      # Using uv (recommended)
      uv sync --group dev
      
      # Or using make
      make setup-dev

3. **Verify Installation**

   .. code-block:: bash

      make verify

Development Workflow
--------------------

1. **Create a Branch**

   .. code-block:: bash

      git checkout -b feature/your-feature-name

2. **Make Changes**

   - Write code following our style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Run Tests and Checks**

   .. code-block:: bash

      # Quick development check
      make dev
      
      # Full CI simulation
      make ci-test

4. **Commit and Push**

   .. code-block:: bash

      git add .
      git commit -m "feat: add your feature description"
      git push origin feature/your-feature-name

5. **Create Pull Request**

   - Open a pull request on GitHub
   - Describe your changes clearly
   - Link to any relevant issues

Code Style Guidelines
---------------------

We use automated tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting  
- **flake8** for linting
- **mypy** for type checking
- **pre-commit** for automated checks

Run formatting:

.. code-block:: bash

   make format

Check code quality:

.. code-block:: bash

   make lint

Testing Guidelines
------------------

All code must be thoroughly tested:

1. **Write Tests**

   - Unit tests for individual functions
   - Integration tests for workflows
   - Use descriptive test names
   - Include edge cases

2. **Test Categories**

   Use pytest markers to categorize tests:

   .. code-block:: python

      @pytest.mark.unit
      def test_spike_list_creation():
          pass

      @pytest.mark.integration  
      def test_full_analysis_pipeline():
          pass

      @pytest.mark.slow
      def test_large_dataset_performance():
          pass

3. **Run Tests**

   .. code-block:: bash

      # All tests
      make test
      
      # Fast tests only
      make test-fast
      
      # With coverage
      make test-coverage

Documentation Guidelines
------------------------

Keep documentation up to date:

1. **Docstrings**

   Use NumPy/Google style docstrings:

   .. code-block:: python

      def compute_metrics(spike_list, config):
          """
          Compute MEA analysis metrics.
          
          Parameters
          ----------
          spike_list : SpikeList
              Input spike data
          config : AnalysisConfig
              Analysis configuration
              
          Returns
          -------
          dict
              Dictionary of computed metrics
          """

2. **Build Documentation**

   .. code-block:: bash

      make docs
      make docs-serve  # View locally

3. **Add Examples**

   Include usage examples in docstrings and tutorials.

Pull Request Guidelines
-----------------------

For a successful pull request:

1. **Before Submitting**

   - ✅ Tests pass locally
   - ✅ Code is formatted and linted
   - ✅ Documentation updated
   - ✅ No merge conflicts

2. **PR Description**

   Include:
   - Clear description of changes
   - Motivation and context
   - Testing performed
   - Breaking changes (if any)

3. **Review Process**

   - Address reviewer feedback
   - Keep discussions respectful
   - Update tests/docs as requested

Types of Contributions
----------------------

**Bug Reports**
   - Use the issue template
   - Include minimal reproduction example
   - Specify environment details

**Feature Requests**
   - Describe the use case
   - Propose API design
   - Consider backward compatibility

**Code Contributions**
   - Follow development workflow
   - Add comprehensive tests
   - Update documentation

**Documentation**
   - Fix typos and errors
   - Add examples and tutorials
   - Improve API documentation

Release Process
---------------

For maintainers:

1. **Prepare Release**

   .. code-block:: bash

      make prepare-release

2. **Update Version**

   Update version in ``pyproject.toml``

3. **Create Release**

   - Tag release: ``git tag v0.1.0``
   - Push tags: ``git push --tags``
   - Create GitHub release
   - CI will automatically publish to PyPI

Development Commands Reference
------------------------------

.. code-block:: bash

   # Setup
   make setup-dev          # Complete development setup
   make install-dev        # Install dev dependencies
   
   # Development
   make dev               # Quick dev check (format + test + verify)
   make format            # Format code
   make lint              # Check code quality
   
   # Testing
   make test              # Run all tests
   make test-fast         # Run fast tests only
   make test-coverage     # Run with coverage
   
   # Documentation
   make docs              # Build documentation
   make docs-serve        # Serve docs locally
   
   # CI Simulation
   make ci-test           # Full CI pipeline
   make ci-quick          # Quick CI check
   
   # Release
   make build             # Build package
   make prepare-release   # Prepare for release

Getting Help
------------

- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check the full documentation
- **Examples**: See the notebooks and examples

Thank you for contributing to MEA-Flow! 🎉