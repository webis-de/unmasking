#!/usr/bin/env python3.6
#
# General-purpose unmasking framework
# Copyright (C) 2017 Janek Bevendorff, Webis Group
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use, copy,
# modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

from conf.loader import JobConfigLoader
from event.dispatch import MultiProcessEventContext
from job.executors import MetaApplyExecutor, MetaEvalExecutor, MetaTrainExecutor
from util.util import SoftKeyboardInterrupt, base_coroutine

from concurrent.futures import ThreadPoolExecutor

import asyncio
import argparse
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

    # apply command
    apply_parser = subparsers.add_parser("apply", help="Apply a trained model to a test dataset.")
    apply_parser.add_argument("model", help="pre-trained input model or labeled raw JSON unmasking data " +
                                            "from which to train a temporary model")
    apply_parser.add_argument("test", help="JSON file containing the raw unmasking data which to classify")
    apply_parser.add_argument("--config", "-c", help="optional job configuration file",
                              required=False, default=None)
    apply_parser.add_argument("--output", "-o", help="output directory to save classification data to",
                              required=False, default=None)

    # eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate the quality of a model against a labeled test set.")
    eval_parser.add_argument("input_train", help="pre-trained input model or labeled raw JSON unmasking data " +
                                                 "from which to train a temporary model")
    eval_parser.add_argument("input_test", help="labeled JSON test set to evaluate the model against")
    eval_parser.add_argument("--config", "-c", help="optional job configuration file",
                             required=False, default=None)
    eval_parser.add_argument("--output", "-o", help="output directory to save evaluation data to",
                             required=False, default=None)

    args = parser.parse_args()

    config_file = args.config

    config_loader = JobConfigLoader(defaults_file="defaults_meta.yml")
    if config_file:
        config_loader.load(config_file)

    loop = asyncio.get_event_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=8))
    try:
        if args.command == "train":
            if not args.input.endswith(".json"):
                print("Input file must be JSON.", file=sys.stderr)
                sys.exit(1)

            executor = MetaTrainExecutor(args.input)
        elif args.command == "apply":
            executor = MetaApplyExecutor(args.model, args.test)
        elif args.command == "eval":
            executor = MetaEvalExecutor(args.input_train, args.input_test)
        else:
            raise ValueError("Invalid sub command")

        future = asyncio.ensure_future(base_coroutine(executor.run(config_loader, args.output)))
        loop.run_until_complete(future)
    finally:
        loop.run_until_complete(base_coroutine(loop.shutdown_asyncgens()))
        loop.stop()
        MultiProcessEventContext.cleanup()


def terminate():
    print("Exited upon user request.", file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, SoftKeyboardInterrupt):
        terminate()
