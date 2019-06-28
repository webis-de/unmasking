# General-purpose Unmasking Framework

This is a general-purpose unmasking framework, primarily (but not only) developed for
authorship verification. For the original unmasking paper, see
[Koppel and Schler (2004)](https://doi.org/10.1145/1015330.1015448).

The framework is released on the [Apache 2.0 license](LICENSE).

## Requirements

The framework needs Python >= 3.6. Dependencies are installed via:

    pip3 install -r requirements.txt

By default, plots are rendered using the `Qt5Agg` Matplotlib backend if run in a graphical
environment or `Agg` otherwise. The graphical backend can be changed by editing the
`matplotlibrc` file (e.g., if you have no Qt on your system).

## Usage

Unmasking experiments are defined in YAML configuration files. The configuration defaults
can be found in `authorship_unmasking/etc/defaults.yml` (for unmasking) and
`authorship_unmasking/etc/defaults_meta.yml` (for the meta classifier). Specific job configurations
can override all or part of these default configurations. An commented example configuration with a
small test corpus can be found in `examples/gutenberg_test`. The default configuration files are also
commented, so please have a look at their contents as well.

### Unmasking

To generate unmasking curve plots, use the `unmask` tool with the `run` command. This will
parse the given job configuration and generate unmasking curves on the provided input data.
The job's output is saved to an `out` folder next to the job configuration.

**Example:**

    ./unmask run examples/gutenberg_test/job.yml

Output will be saved to `examples/gutenberg_test/out`.

The output consists of individual curve plots generated with various parameters, the
raw numbers as JSON, as well as an average aggregation of all individual parameter
configurations (see the example job configuration file for more details). If you only want
to aggregate existing runs, use

    ./unmask aggregate JSON_FILE [JSON_FILE ...]

where `JSON_FILE` is the generated raw JSON file of one or more existing runs.

For a full list of all parameters, specify the `-h` flag:

    ./unmask run -h
    ./unmask aggregate -h

### Meta Classification

After you have generated your unmasking curves, you can train and evaluate meta classification
models on them by using the `classify` tool. It comes with the `train`, `apply`, `eval`, and
`model_select` commands for training a new model, applying a previously-trained model,
evaluating a model on a corpus with a ground truth, and selecting the best-performing model
of a series of pre-trained models.

The `classify` tool also takes a YAML job configuration via the `--config` flag, although
in most cases it shouldn't be necessary to write a custom configuration. If no configuration
is given, the defaults from `authorship_unmasking/etc/defaults_meta.yml` will be used.

**Examples:**

Train a model on the JSON output of an unmasking job:

    ./classify train INPUT_JSON

Apply a pre-trained model on unlabeled unmasking JSON:

    ./classify apply MODEL INPUT_JSON

Train and evaluate a classifier an two labeled unmasking JSON dumps:

    ./classify eval INPUT_JSON_TRAIN INPUT_JSON_TEST

Select the best-performing model from the input directory (one sub directory per model):

    ./classify model_select INPUT_DIR

For a full list of all parameters, use the help parameter `-h` on a command.
