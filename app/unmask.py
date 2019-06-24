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

from authorship_unmasking.conf.loader import JobConfigLoader
from authorship_unmasking.job.executors import AggregateExecutor, ExpandingExecutor
from authorship_unmasking.output.formats import UnmaskingResult
from authorship_unmasking.util.util import SoftKeyboardInterrupt, run_in_event_loop

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="unmasking",
        description="Run unmasking jobs.",
        add_help=True)

    subparsers = parser.add_subparsers(title="commands", dest="command")
    subparsers.required = True

    # run command
    run_parser = subparsers.add_parser("run", help="Run unmasking job.")
    run_parser.add_argument("config", metavar="CONFIG", type=str, help="job configuration")
    run_parser.add_argument("-o", "--output", metavar="DIR", type=str, default=None,
                            help="override output directory")
    run_parser.add_argument("-w", "--wait", action="store_true",
                            help="wait for user confirmation after job is done")

    # aggregate command
    agg_parser = subparsers.add_parser("aggregate", help="Aggregate existing run outputs.")
    agg_parser.add_argument("input", help="input JSON files", nargs="+")
    agg_parser.add_argument("--config", "-c", help="optional job configuration file",
                            required=False, default=None)
    agg_parser.add_argument("--output", "-o", help="output directory to save the trained model to",
                            required=False, default=None)
    agg_parser.add_argument("--wait", "-w", help="wait for user confirmation after job is done",
                            required=False, action="store_true")

    args = parser.parse_args()

    config_loader = JobConfigLoader(defaults_file="defaults.yml")
    if args.config:
        if not os.path.exists(args.config):
            print("ERROR: Configuration file '{}' does not exist.".format(args.config), file=sys.stderr)
            sys.exit(1)

        config_loader.load(args.config)

    if args.command == "run":
        executor = ExpandingExecutor()

    elif args.command == "aggregate":
        unmasking_results = []
        for input_file in args.input:
            assert_file(input_file)
            if not input_file.endswith(".json"):
                print("Input file must be JSON.", file=sys.stderr)
                sys.exit(1)

            res = UnmaskingResult()
            res.load(input_file)
            unmasking_results.append(res)

        executor = AggregateExecutor(unmasking_results)

    else:
        raise RuntimeError("Invalid sub command: {}".format(args.command))

    run_in_event_loop(executor, config_loader, args.output)

    if args.wait:
        input("Press enter to terminate...")


def terminate():
    print("Exited upon user request.", file=sys.stderr)
    sys.exit(1)


def assert_file(file_name: str):
    if not os.path.isfile(file_name):
        print("File '{}' does not exist.".format(file_name), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, SoftKeyboardInterrupt):
        terminate()
