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
from job.executors import ExpandingExecutor

from concurrent.futures import ThreadPoolExecutor

import asyncio
import argparse
import os
import sys


class SoftKeyboardInterrupt(Exception):
    """
    Replacement for KeyboardInterrupt that inherits from :class:: Exception instead of
    :class:: BaseException, to avoid uncatchable stack traces when a :class:: KeyboardInterrupt
    happens within a coroutine.
    See: https://github.com/python/asyncio/issues/341
    """
    pass


async def base_coroutine(cr):
    """
    Base coroutine that wraps and waits another coroutine and catches KeyboardInterrupts.
    Caught keyboardInterrupts are re-raised as SoftKeyboardInterrupts.

    :param cr: coroutine to wrap
    :return: Return value of the wrapped coroutine
    """
    try:
        return await cr
    except KeyboardInterrupt as k:
        raise SoftKeyboardInterrupt() from k


def main():
    parser = argparse.ArgumentParser(
        prog="unmasking",
        description="Run unmasking jobs.",
        add_help=True)
    parser.add_argument("config",
                        metavar="CONFIG",
                        type=str,
                        help="job configuration")
    parser.add_argument("-o", "--output",
                        metavar="DIR",
                        type=str,
                        default=None,
                        help="override output directory")
    parser.add_argument("-w", "--wait",
                        action="store_true",
                        help="wait for user confirmation after job is done")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print("ERROR: Configuration file '{}' does not exist.".format(args.config), file=sys.stderr)
        sys.exit(1)

    config_loader = JobConfigLoader()
    config_loader.load(args.config)

    loop = asyncio.get_event_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=8))
    try:
        executor = ExpandingExecutor()
        future = asyncio.ensure_future(base_coroutine(executor.run(config_loader, args.output)))
        loop.run_until_complete(future)
    finally:
        loop.run_until_complete(base_coroutine(loop.shutdown_asyncgens()))
        loop.stop()
        MultiProcessEventContext.cleanup()

    if args.wait:
        input("Press enter to terminate...")


def terminate():
    print("Exited upon user request.", file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, SoftKeyboardInterrupt):
        terminate()
