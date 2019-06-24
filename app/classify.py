# Copyright (C) 2017-2019 Janek Bevendorff, Webis Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from authorship_unmasking.job.executors import *
from authorship_unmasking.util.util import SoftKeyboardInterrupt, run_in_event_loop

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="classify",
        description="Train and apply unmasking models.",
        add_help=True)

    subparsers = parser.add_subparsers(title="commands", dest="command")
    subparsers.required = True

    # train command
    train_parser = subparsers.add_parser("train", help="Train a model from unmasking curves.")
    train_parser.add_argument("input", help="labeled JSON training set for which to build the model")
    train_parser.add_argument("--config", "-c", help="optional job configuration file",
                              required=False, default=None)
    train_parser.add_argument("--output", "-o", help="output directory to save the trained model to",
                              required=False, default=None)
    train_parser.add_argument("--wait", "-w", help="wait for user confirmation after job is done",
                              required=False, action="store_true")

    # apply command
    apply_parser = subparsers.add_parser("apply", help="Apply a trained model to a test dataset.")
    apply_parser.add_argument("model", help="pre-trained input model or labeled raw JSON unmasking data " +
                                            "from which to train a temporary model")
    apply_parser.add_argument("test", help="JSON file containing the raw unmasking data which to classify")
    apply_parser.add_argument("--config", "-c", help="optional job configuration file",
                              required=False, default=None)
    apply_parser.add_argument("--output", "-o", help="output directory to save classification data to",
                              required=False, default=None)
    apply_parser.add_argument("--wait", "-w", help="wait for user confirmation after job is done",
                              required=False, action="store_true")

    # eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate the quality of a model against a labeled test set.")
    eval_parser.add_argument("input_train", help="pre-trained input model or labeled raw JSON unmasking data " +
                                                 "from which to train a temporary model")
    eval_parser.add_argument("input_test", help="labeled JSON test set to evaluate the model against")
    eval_parser.add_argument("--config", "-c", help="optional job configuration file",
                             required=False, default=None)
    eval_parser.add_argument("--output", "-o", help="output directory to save evaluation data to",
                             required=False, default=None)
    eval_parser.add_argument("--wait", "-w", help="wait for user confirmation after job is done",
                             required=False, action="store_true")

    # model_select command
    eval_parser = subparsers.add_parser("model_select",
                                        help="Select the best-performing unmasking model of a set of configurations.")
    eval_parser.add_argument("input_run_folder", help="folder containing the unmasking runs from whose configurations" +
                                                      "to select the best performing model")
    eval_parser.add_argument("--config", "-c", help="optional job configuration file",
                             required=False, default=None)
    eval_parser.add_argument("--output", "-o", help="output directory to save evaluation data to",
                             required=False, default=None)
    eval_parser.add_argument("--cv_folds", "-f", help="cross-validation folds for model selection",
                             required=False, type=int, default=10)
    eval_parser.add_argument("--wait", "-w", help="wait for user confirmation after job is done",
                             required=False, action="store_true")

    args = parser.parse_args()

    config_loader = JobConfigLoader(defaults_file="defaults_meta.yml")
    if args.config:
        config_loader.load(args.config)

    if args.command == "train":
        assert_file(args.input)
        if not args.input.endswith(".json"):
            print("Input file must be JSON.", file=sys.stderr)
            sys.exit(1)

        executor = MetaTrainExecutor(args.input)
    elif args.command == "apply":
        assert_file(args.model)
        assert_file(args.test)
        executor = MetaApplyExecutor(args.model, args.test)
    elif args.command == "eval":
        assert_file(args.input_train)
        assert_file(args.input_test)
        executor = MetaEvalExecutor(args.input_train, args.input_test)
    elif args.command == "model_select":
        assert_dir(args.input_run_folder)
        executor = MetaModelSelectionExecutor(args.input_run_folder, args.cv_folds)
    else:
        raise RuntimeError("Invalid sub command: {}".format(args.command))

    run_in_event_loop(executor.run(config_loader, args.output))

    if args.wait:
        input("Press enter to terminate...")


def assert_file(file_name: str):
    if not os.path.isfile(file_name):
        print("File '{}' does not exist.".format(file_name), file=sys.stderr)
        sys.exit(1)


def assert_dir(dirname: str):
    if not os.path.isdir(dirname):
        print("Directory '{}' does not exist or is not a directory.".format(dirname), file=sys.stderr)
        sys.exit(1)


def terminate():
    print("Exited upon user request.", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, SoftKeyboardInterrupt):
        terminate()
