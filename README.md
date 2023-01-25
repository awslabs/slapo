# Sphinx-Gallery template

This is a template for using [Sphinx](https://www.sphinx-doc.org/en/master/)
and [Sphinx-Gallery](https://sphinx-gallery.github.io/stable/index.html)
to document a Python package with examples. It includes:

* a sample package (`SampleModule`) with two modules:
    * `module.py` which defines an example class and an example function
    * `data_download.py` which provides functions for downloading large
      datasets required for examples and storing the path to the downloaded
      datasets to ensure that they are only downloaded once
* two galleries of examples that demonstrate Sphinx-Gallery functions
* a way to automatically build the documentation whenever the repository is
  pushed to

# Quick start

1. Clone this repository:
    ```
    git clone https://github.com/sphinx-gallery/sample-project.git
    ```
2. Update the `SampleModule` to include your package modules instead
3. Update the `docs/conf.py` with your project details in this section:
    ```
    # -- Project information -----------------------------------------------------

    project = 'sample-project'
    copyright = '2020, <Author>'
    author = '<Author>'
    # The full version, including alpha/beta/rc tags
    release = '0.0.1'
    ```
   and desired gallery configurations in this section:
    ```
    # sphinx-gallery configuration
    sphinx_gallery_conf = {
        # path to your example scripts
        'examples_dirs': ['../sample-gallery-1', '../sample-gallery-2'],
        # path to where to save gallery generated output
        'gallery_dirs': ['auto_gallery-1', 'auto_gallery-2'],
        # specify that examples should be ordered according to filename
        'within_subsection_order': FileNameSortKey,
        # directory where function granular galleries are stored
        'backreferences_dir': 'gen_modules/backreferences',
        # Modules for which function level galleries are created.  In
        # this case sphinx_gallery and numpy in a tuple of strings.
        'doc_module': ('SampleModule'),
    }
    ```
4. Update `sample-gallery-1` and `sample-gallery-2` to your own example
   galleries. Use the command:
    ```
    $ make html
    ```
   from the directory `docs/` to build the documentation and see the effects of
   your changes to the example `.py` files.

# Sphinx & Sphinx-Gallery configurations

For full configuration options see [Sphinx documentation](https://www.sphinx-doc.org/en/master/usage/configuration.html) and
[Sphinx-Gallery documentaton](https://sphinx-gallery.github.io/stable/configuration.html)

# Data downloading

If your examples require datasets (especially if the same dataset is used
across several examples), you may wish to use the functions
defined in `SampleModule/data_download.py` to download the required datasets
during building of the documentation. The `download_data` function first checks
if the data file already exists in either the path saved under the key
``data_key`` in the config file or the default data path;
``~/sg_template_data``. If the file does not exist, it downloads the data
from ``url`` argument given and saves to a filename specified in the argument
``data_file_name``. Finally, it also stores the location of the data in a
configuration file, under key ``data_key``. See
`sample-gallery-1/plot_1_download_data.py` for an example of the function in
use.

The following projects use a similar method to make datasets available when
building their examples and can be looked to for guidance, especially if your
project has specific data needs:
* [MNE-python](https://github.com/mne-tools/mne-python), specifically
  [dataset/utils](https://github.com/mne-tools/mne-python/blob/master/mne/datasets/utils.py)
* [scikit-learn](https://github.com/scikit-learn/scikit-learn), specifically
  [datasets/_base](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/datasets/_base.py#L39-L41)

# Automatic document building

Configuration templates are provided for using [CircleCI](https://circleci.com/)
and [Github actions](https://help.github.com/en/actions) to automatically
build your documentation after every push. This allows you to view changes
in the built documentation and be informed of any errors during building.

Both options are discussed in detail below to help you decide which is more
suitable for your project.

## CircleCI

To use CircleCI in your project, first follow their sign up steps
[here](https://circleci.com/docs/2.0/first-steps/). The `.circleci/config.yml`
file outlines a simple workflow that builds the documentation and stores
it as an artifact after every push. If you wish to make changes to the workflow,
see [CircleCI configuration](https://circleci.com/docs/2.0/config-intro/)
to learn how to amend the `.circleci/config.yml` for
your project needs. Additionally, to enable CircleCI builds on forked
pull requests, follow the steps [here](https://circleci.com/docs/2.0/oss/#build-pull-requests-from-forked-repositories).

A handy link to the buit documentation is also added to Github via the
Github action [circleci-artifacts-redirector](https://github.com/larsoner/-artifacts-redirector-action),
via the configuration file `.github/workflows/main.yml`.

## Github actions

The workflow stored at `.github/workflows/sphinx_build.yml` uses the community
action [Sphinx build action](https://github.com/ammaraskar/sphinx-action)
to build the documentation. The Github action [upload artifact](https://github.com/actions/upload-artifact)
is then used to store the built documentation. The built documentation can
be downloaded by navigating to the 'Actions' tab of the repository but not
be directly viewed online like with CircleCI. There are also
limitations on how long the artifact is stored, see [here](https://help.github.com/en/actions/configuring-and-managing-workflows/persisting-workflow-data-using-artifacts) for
more details. Additonally, if you wish to make changes to the workflow see
[here](https://help.github.com/en/actions/configuring-and-managing-workflows)
to learn how to amend the `.github/workflows/sphinx_build.yml` file for your
project needs.

Finally, it may be useful to [enable debug logging](https://help.github.com/en/actions/configuring-and-managing-workflows/managing-a-workflow-run#enabling-debug-logging)
to aid in diagnosing why your workflow is not working.
