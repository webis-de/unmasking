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
    train_parser.add_argument("output", help="output file to save the model to")
    train_parser.add_argument("-t", "--threshold", help="discard samples below this uncertainty threshold",
                              type=float, required=False, default=0.0, metavar="NUM")

    # apply command
    apply_parser = subparsers.add_parser("apply", help="Apply a trained model to a test dataset.")
    apply_parser.add_argument("model", help="file containing the trained model")
    apply_parser.add_argument("-o", "--output", help="output file (default: print to STDOUT)",
                              required=False, default=None, metavar="FILE")
    apply_parser.add_argument("-t", "--threshold", help="discard samples below this uncertainty threshold",
                              type=float, required=False, default=0.0, metavar="NUM")
    apply_parser.add_argument("-p", "--plot-file", help="save plot of filtered curves to file",
                              required=False, default=None, metavar="FILE")
    apply_parser.add_argument("-r", "--plot-rc", help="plot RC file",
                              required=False, default="etc/plot_rc.yml", metavar="FILE")
    apply_parser.add_argument("-d", "--display", help="display plot on screen",
                              required=False, default=False, action="store_true")

    # eval command
    eval_parser = subparsers.add_parser("eval", help="Train and evaluate a model based on labeled test and " +
                                                     "training sets without saving the model.")
    eval_parser.add_argument("input_train", help="labeled JSON training set for which to build the model")
    eval_parser.add_argument("input_test", help="labeled JSON test set to evaluate the model against")
    eval_parser.add_argument("-o", "--output", help="output file (default: print to STDOUT)",
                             required=False, default=None, metavar="FILE")
    eval_parser.add_argument("-t", "--threshold", help="discard samples below this uncertainty threshold",
                             type=float, required=False, default=0.0, metavar="NUM")
    eval_parser.add_argument("-p", "--plot-file", help="save plot of filtered curves to file",
                             required=False, default=None, metavar="FILE")
    eval_parser.add_argument("-r", "--plot-rc", help="plot RC file",
                             required=False, default="etc/plot_rc.yml", metavar="FILE")
    eval_parser.add_argument("-d", "--display", help="display plot on screen",
                             required=False, default=False, action="store_true")

    args = parser.parse_args()
    if args.command == "train":
        pass
    elif args.command == "apply":
        pass
    elif args.command == "eval":
        pass


def terminate():
    print("Exited upon user request.", file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        terminate()
