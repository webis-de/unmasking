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

from event.dispatch import EventBroadcaster
from event.events import PairBuildingProgressEvent, PairChunkingProgressEvent
from input.interfaces import SamplePair, SamplePairClass, Tokenizer
from input.interfaces import CorpusParser

import math
import os
import random
import re
import xml.etree.ElementTree as etree
from glob import glob
from itertools import combinations, combinations_with_replacement
from typing import Callable, AsyncGenerator, List, Optional
from urllib.parse import urlparse
from uuid import uuid5


class SamplePairImpl(SamplePair):
    """
    Concrete SamplePair implementation.

    Events published by this class:

    * `onChunkingProgress`: [type PairChunkingProgressEvent]
                            fired during chunk generation to indicate progress
    """

    def __init__(self, cls: SamplePairClass, chunk_tokenizer: Tokenizer):
        super().__init__(cls, chunk_tokenizer)

        self._pair_id = None
        self._chunks_a = []
        self._chunks_b = []
        self._progress_event = None
        self._a = None
        self._b = None

    async def chunk(self, a: List[str], b: List[str]):
        self._a = a
        self._b = b

        group_id = PairChunkingProgressEvent.generate_group_id([self.pair_id])
        total_events = len(a) + len(b)
        self._progress_event = PairChunkingProgressEvent(group_id, 0, total_events)

        await EventBroadcaster.publish("onChunkingProgress", self._progress_event, self.__class__.__bases__[0])

        for text in a:
            async for tokens in self._chunk_tokenizer.await_tokens(text):
                self._chunks_a.append(tokens)
            self._progress_event = PairChunkingProgressEvent.new_event(self._progress_event)
            await EventBroadcaster.publish("onChunkingProgress", self._progress_event, self.__class__.__bases__[0])

        for text in b:
            async for tokens in self._chunk_tokenizer.await_tokens(text):
                self._chunks_b.append(tokens)
            self._progress_event = PairChunkingProgressEvent.new_event(self._progress_event)
            await EventBroadcaster.publish("onChunkingProgress", self._progress_event, self.__class__.__bases__[0])

    @property
    def cls(self) -> type:
        return self._cls

    @property
    def pair_id(self) -> Optional[str]:
        if self._pair_id is None:
            self._pair_id = str(uuid5(self.SAMPLE_PAIR_NS, "\n".join(sorted(self._a) + sorted(self._b))))

        return self._pair_id

    @property
    def chunks_a(self) -> List[str]:
        return self._chunks_a

    @property
    def chunks_b(self) -> List[str]:
        return self._chunks_b


class TextPairParser(CorpusParser):
    """
    Parser for generating all possible combinations of text pairs and labeling them according
    to whether they were written by the same author or not.

    Expects a directory structure where there is one folder for each author containing
    at least two samples of their work.

    Example:

        + Ernest_Hemingway
        |__ + The_Torrents_of_Spring.txt
        |__ + Islands_in_the_Stream.txt
        |__ + The_Garden_of_Eden.txt
        + William_Faulkner
        |__ + Soldier's_Pay.txt
        |__ + Light_in_August.txt

    File and folder names can be chosen arbitrarily, but the book sample files must end in .txt.

    Events published by this class:

    * `onPairGenerated`: [type PairBuildingProgressEvent]
                         fired when a pair has been generated
    """

    class Class(SamplePairClass):
        UNSPECIFIED = -1
        DIFFERENT_AUTHORS = 0
        SAME_AUTHOR = 1

    def __init__(self, chunk_tokenizer: Tokenizer, corpus_path: str = None):
        super().__init__(chunk_tokenizer, corpus_path)

        self._input_files = {}
        self._input_authors = {}
        self._is_prepared = False

    async def _prepare(self):
        """
        Prepare file and author lists to be parsed.
        """
        if self._is_prepared:
            return

        if not os.path.isdir(self.corpus_path):
            raise IOError("Corpus '{}' not found".format(self.corpus_path))

        dirs = os.listdir(self.corpus_path)
        for d in dirs:
            dir_path = os.path.realpath(os.path.join(self.corpus_path, d))
            if not os.path.isdir(dir_path):
                continue

            files = sorted(os.listdir(dir_path))
            for f in files:
                file_path = os.path.realpath(os.path.join(dir_path, f))
                if not os.path.isfile(file_path) or not f.endswith(".txt"):
                    continue

                self._input_files[file_path] = d
                if d not in self._input_authors:
                    self._input_authors[d] = []
                self._input_authors[d].append(file_path)

        self._is_prepared = True

    async def __aiter__(self) -> AsyncGenerator[SamplePair, None]:
        await self._prepare()

        num_combinations = math.factorial(len(self._input_files)) // 2 // math.factorial(len(self._input_files) - 2)
        pair_num = 0

        for f1, f2 in combinations(self._input_files.keys(), 2):
            f1_contents = await self.await_file(f1)
            f2_contents = await self.await_file(f2)

            cls = self.Class.SAME_AUTHOR if self._input_files[f1] == self._input_files[f2] \
                else self.Class.DIFFERENT_AUTHORS
            pair = SamplePairImpl(cls, self.chunk_tokenizer)
            await pair.chunk([f1_contents], [f2_contents])

            group_id = PairBuildingProgressEvent.generate_group_id(["a:" + f1] + ["b:" + f2])
            await EventBroadcaster.publish("onPairGenerated",
                                           PairBuildingProgressEvent(group_id, pair_num, num_combinations,
                                                                     pair, [f1], [f2]),
                                           self.__class__)
            yield pair
            pair_num += 1


class AuthorPairParser(TextPairParser):
    """
    Parser that takes each text from every author and generates pairs with collections of
    all texts of all other authors and the collection of all other texts of the same author.

    Expected input directory structure and published events are the same
    as for :class:: TextPairParser.
    """

    async def __aiter__(self) -> AsyncGenerator[SamplePair, None]:
        await self._prepare()

        pair_num = 0
        single_file_sets = []

        for a1, a2 in combinations_with_replacement(self._input_authors.keys(), 2):
            for f1 in self._input_authors[a1]:
                f2 = sorted([f for f in self._input_authors[a2] if f != f1])
                if not f2:
                    # skip if author has only one file
                    continue

                if len(f2) == 1:
                    fs = {f1, f2[0]}
                    if fs in single_file_sets:
                        # We already compared these two texts
                        continue
                    single_file_sets.append(fs)

                f1_contents = await self.await_file(f1)
                f2_contents = [await self.await_file(f) for f in f2]

                cls = self.Class.SAME_AUTHOR if a1 == a2 else self.Class.DIFFERENT_AUTHORS
                pair = SamplePairImpl(cls, self.chunk_tokenizer)
                await pair.chunk([f1_contents], f2_contents)

                group_id = PairBuildingProgressEvent.generate_group_id(["a:" + f1] + ["b:" + ",".join(f2)])
                await EventBroadcaster.publish("onPairGenerated",
                                               PairBuildingProgressEvent(group_id, pair_num, None,
                                                                         pair, [f1], f2),
                                               self.__class__)

                yield pair
                pair_num += 1


class WebisBuzzfeedAuthorshipCorpusParser(CorpusParser):
    """
    Corpus parser for the Webis BuzzFeed corpus.
    This parser is intended for building pairs of texts by individual or portal authorship.
    For classifying by categories such as political orientation use :class:: BuzzFeedXMLCorpusParser.

    Pairs are generated by randomly drawing texts from the input set without replacement.

    Events published by this class:

    * `onPairGenerated`: [type PairBuildingProgressEvent]
                         fired when a pair has been generated
    """

    class Class(SamplePairClass):
        UNSPECIFIED = -1
        SAME_PORTAL = 0
        DIFFERENT_PORTALS = 1

    def __init__(self, corpus_path: str, chunk_tokenizer: Tokenizer, datasets: List[str], samples: int = 100):
        """
        :param datasets: datasets within the corpus to parse
        :param samples: number of samples to draw per class. If the actual number of samples of a single class
                        is less than half of this, additional samples will be generated by random oversampling.
                        If both classes ave less then half this number of samples, they will be skipped.
        """
        super().__init__(chunk_tokenizer, corpus_path)
        self._datasets = datasets
        self._samples = samples

    async def __aiter__(self) -> AsyncGenerator[SamplePair, None]:
        texts_by_portals = {}

        for ds in self._datasets:
            ds_path = os.path.join(self.corpus_path, ds)
            files = os.listdir(ds_path)

            for f in files:
                file_path = os.path.join(ds_path, f)

                if not os.path.isfile(file_path) or not f.endswith(".xml"):
                    continue

                xml = etree.parse(file_path).getroot()
                portal = ""
                main_text = ""
                done = 0
                for e in xml:
                    if e.tag == "uri":
                        portal = urlparse(str(e.text)).hostname
                        done += 1
                    if e.tag == "mainText":
                        main_text = str(e.text)
                        done += 1
                    if done >= 2:
                        break

                if portal == "" or main_text == "":
                    continue

                if portal not in texts_by_portals:
                    texts_by_portals[portal] = []
                texts_by_portals[portal].append((file_path, main_text))

        # discard all portals with too few texts
        discard = []
        for p in texts_by_portals:
            if len(texts_by_portals[p]) < 50:
                discard.append(p)
        texts_by_portals = {k: v for (k, v) in texts_by_portals.items() if k not in discard}

        pair_num = 0

        for cls1 in texts_by_portals:
            num_texts1 = len(texts_by_portals[cls1])

            for cls2 in texts_by_portals:
                num_texts2 = len(texts_by_portals[cls2])

                # number of already matched chunk / text pairs
                pair_counter = 0

                # keep track of already drawn texts
                drawn_a = []
                drawn_b = []

                # final chunks of a pair
                chunks_a = []
                chunks_b = []

                # file names of drawn texts
                file_names_a = []
                file_names_b = []

                while pair_counter < self._samples and len(drawn_a) < num_texts1 and len(drawn_b) < num_texts2:
                    idx1 = random.randint(0, num_texts1 - 1)
                    idx2 = random.randint(0, num_texts2 - 1)
                    if cls1 == cls2:
                        # make sure the cut between both sets is always empty
                        # when comparing a class against itself
                        if idx1 == idx2:
                            continue
                        if idx1 in drawn_a or idx1 in drawn_b:
                            continue
                        if idx2 in drawn_a or idx2 in drawn_b:
                            continue

                    if idx1 in drawn_a or idx2 in drawn_b:
                        continue

                    chunks_a.append(texts_by_portals[cls1][idx1][1])
                    chunks_b.append(texts_by_portals[cls2][idx2][1])
                    file_names_a.append(texts_by_portals[cls1][idx1][0])
                    file_names_b.append(texts_by_portals[cls2][idx2][0])
                    drawn_a.append(idx1)
                    drawn_b.append(idx2)

                    pair_counter += 1

                    # break earlier when we are comparing a class with itself, since
                    # we only need half the number of iterations
                    if cls1 == cls2 and len(drawn_a) >= num_texts1 // 2:
                        break

                    # generate more samples by random oversampling when one class has less
                    # than self._samples // 2 samples
                    if pair_counter < self._samples // 2 and len(drawn_a) >= num_texts1:
                        drawn_a = []
                    elif pair_counter < self._samples // 2 and len(drawn_b) >= num_texts2:
                        drawn_b = []

                pair_class = self.Class.DIFFERENT_PORTALS
                if cls1 == cls2:
                    pair_class = self.Class.SAME_PORTAL

                pair = SamplePairImpl(pair_class, self.chunk_tokenizer)
                await pair.chunk(chunks_a, chunks_b)
                group_id = PairBuildingProgressEvent.generate_group_id([pair.pair_id])
                await EventBroadcaster.publish("onPairGenerated",
                                               PairBuildingProgressEvent(group_id, pair_num, None,
                                                                         pair, file_names_a, file_names_b),
                                               self.__class__)
                pair_num += 1
                yield pair


class WebisBuzzfeedCatCorpusParser(CorpusParser):
    """
    Corpus parser for the Webis BuzzFeed corpus.
    This parser is intended for building pairs of texts by categories such as political orientation or
    veracity. For discriminating texts by authorship, use :class:`BuzzFeedAuthorshipXMLCorpusParser`.

    Pairs are generated by randomly drawing texts from the input set without replacement.

    * `onPairGenerated`:    [type PairBuildingProgressEvent]
                            fired when a pair has been generated
    """

    class PairClass(SamplePairClass):
        UNSPECIFIED = -1

        LEFT_LEFT = 0
        RIGHT_RIGHT = 1
        MAINSTREAM_MAINSTREAM = 2
        LEFT_RIGHT = 3
        LEFT_MAINSTREAM = 4
        RIGHT_MAINSTREAM = 5

        FAKE_FAKE = 10
        REAL_REAL = 11
        SATIRE_SATIRE = 12
        FAKE_REAL = 13
        FAKE_SATIRE = 14
        SATIRE_REAL = 15

        FAKE_LEFT_FAKE_RIGHT = 20
        FAKE_LEFT_REAL_LEFT = 21
        FAKE_RIGHT_REAL_RIGHT = 22
        REAL_RIGHT_REAL_LEFT = 23

    class SingleTextClass(SamplePairClass):
        UNSPECIFIED = -1

        LEFT = 0
        RIGHT = 1
        MAINSTREAM = 2

        SATIRE = 10
        FAKE = 11
        REAL = 12

        FAKE_LEFT = 20
        FAKE_RIGHT = 21
        REAL_LEFT = 22
        REAL_RIGHT = 23

    def __init__(self, corpus_path: str, chunk_tokenizer: Tokenizer, datasets: List[str],
                 class_assigner: Callable[[etree.Element], SingleTextClass], samples: int = 100):
        """
        :param datasets: datasets within the corpus to parse
        :param class_assigner: callable object to assign proper class to a document
                               (provided pre-defined methods: :method:`class_by_orientation()`,
                               :method:`class_by_veracity()`, :method:`class_by_orientation_and_veracity()`)
        :param samples: number of samples to draw per class. If the actual number of samples of a single class
                        is less than half of this, additional samples will be generated by random oversampling.
                        If both classes ave less then half this number of samples, they will be skipped.
        """
        super().__init__(chunk_tokenizer, corpus_path)
        self._datasets = datasets
        self._class_assigner = class_assigner
        self._samples = samples

    @staticmethod
    def class_by_orientation(xmlroot: etree.Element) -> SingleTextClass:
        """
        Assign class to pair based on orientation. Assigns classes LEFT, RIGHT or MAINSTREAM.
        Class will be UNSPECIFIED when texts don't match any of these classes.
        Use a reference to this method as parameter for the constructor.

        :param xmlroot: XML root of the text
        :return: assigned class
        """
        cls = None

        for c in xmlroot:
            if c.tag == "orientation":
                cls = c.text
                break

        e = WebisBuzzfeedCatCorpusParser.SingleTextClass
        if cls == "left":
            return e.LEFT
        elif cls == "right":
            return e.RIGHT
        elif cls == "mainstream":
            return e.MAINSTREAM

        return e.UNSPECIFIED

    @staticmethod
    def class_by_veracity(xmlroot: etree.Element) -> SingleTextClass:
        """
        Assign class to pair based on orientation. Assigns classes SATIRE, FAKE or REAL.
        Class will be UNSPECIFIED when texts don't match any of these classes.
        Use a reference to this method as parameter for the constructor.

        :param xmlroot: XML root of the text
        :return: assigned class
        """
        cls = None

        for c in xmlroot:
            if c.tag == "orientation" and c.text == "satire":
                cls = "satire"
                break
            if c.tag == "veracity":
                cls = c.text
                # don't break to make sure satire overrides veracity

        e = WebisBuzzfeedCatCorpusParser.SingleTextClass
        if cls == "satire":
            return e.SATIRE
        elif cls == "mostly false" or cls == "mixture of true and false":
            return e.FAKE
        elif cls == "mostly true":
            return e.REAL

        return e.UNSPECIFIED

    @staticmethod
    def class_by_orientation_and_veracity(xmlroot: etree.Element) -> SingleTextClass:
        """
        Assign class to pair based on orientation. Assigns classes SATIRE, FAKE or REAL.
        Class will be UNSPECIFIED when texts don't match any of these classes.
        Use a reference to this method as parameter for the constructor.

        :param xmlroot: XML root of the text
        :return: assigned class
        """
        ver = None
        ori = None

        done = 0
        for c in xmlroot:
            if c.tag == "veracity":
                ver = c.text
                done += 1
            if c.tag == "orientation":
                ori = c.text
                done += 1
            if done >= 2:
                break

        fake = (ver == "mostly false" or ver == "mixture of true and false")
        real = (ver == "mostly true")

        e = WebisBuzzfeedCatCorpusParser.SingleTextClass
        if fake and ori == "left":
            return e.FAKE_LEFT
        elif fake and ori == "right":
            return e.FAKE_RIGHT
        elif real and ori == "left":
            return e.REAL_LEFT
        elif real and ori == "right":
            return e.REAL_RIGHT

        return e.UNSPECIFIED

    async def __aiter__(self) -> AsyncGenerator[SamplePair, None]:
        texts_by_class = {}
        for ds in self._datasets:
            ds_path = os.path.join(self.corpus_path, ds)
            files = os.listdir(ds_path)

            for f in files:
                file_path = os.path.join(ds_path, f)

                if not os.path.isfile(file_path) or not f.endswith(".xml"):
                    continue

                xml = etree.parse(file_path).getroot()
                cls = self._class_assigner(xml)
                if cls == self.SingleTextClass.UNSPECIFIED:
                    continue

                if cls not in texts_by_class:
                    texts_by_class[cls] = []
                texts_by_class[cls].append((file_path, xml))

        # compound classes to build
        processed_comp_classes = []

        pair_num = 0

        for cls1 in texts_by_class:
            num_texts1 = len(texts_by_class[cls1])

            for cls2 in texts_by_class:
                pair_class = None
                try:
                    pair_class = self.PairClass[str(cls1) + "_" + str(cls2)]
                    pair_class = self.PairClass[str(cls2) + "_" + str(cls1)]
                except KeyError:
                    if pair_class is None:
                        continue

                comp_class = {cls1, cls2}
                if comp_class in processed_comp_classes:
                    continue
                processed_comp_classes.append(comp_class)

                num_texts2 = len(texts_by_class[cls2])

                # list to keep track of already drawn texts, so we don't use them again
                drawn_a = []
                drawn_b = []

                # number of already matched chunk / text pairs
                pair_counter = 0

                # final chunks of a pair
                chunks_a = []
                chunks_b = []

                # file names of drawn texts
                file_names_a = []
                file_names_b = []

                # skip if both classes have too few samples
                if num_texts1 + num_texts2 < self._samples or \
                   cls1 == cls2 and num_texts1 < self._samples // 4:
                    continue

                while pair_counter < self._samples and len(drawn_a) < num_texts1 and len(drawn_b) < num_texts2 > 0:
                    idx1 = random.randint(0, num_texts1 - 1)
                    idx2 = random.randint(0, num_texts2 - 1)
                    if cls1 == cls2:
                        # make sure the cut between both sets is always empty
                        # when comparing a class against itself
                        if idx1 == idx2:
                            continue
                        if idx1 in drawn_a or idx1 in drawn_b:
                            continue
                        if idx2 in drawn_a or idx2 in drawn_b:
                            continue

                    if idx1 in drawn_a or idx2 in drawn_b:
                        continue

                    for e in texts_by_class[cls1][idx1][1]:
                        if e.tag == "mainText":
                            chunks_a.append(str(e.text))
                            file_names_a.append(texts_by_class[cls1][idx1][0])
                            break
                    drawn_a.append(idx1)
                    for e in texts_by_class[cls2][idx2][1]:
                        if e.tag == "mainText":
                            chunks_b.append(str(e.text))
                            file_names_b.append(texts_by_class[cls2][idx2][0])
                            break
                    drawn_b.append(idx2)

                    pair_counter += 1

                    # break earlier when we are comparing a class with itself, since
                    # we only need half the number of iterations
                    if cls1 == cls2 and len(drawn_a) >= num_texts1 // 2:
                        break

                    # generate more samples by random oversampling when one class has less
                    # than self._samples // 2 samples
                    #if pair_counter < self._samples // 2 and len(drawn_a) >= num_texts1:
                    #    drawn_a = []
                    #elif pair_counter < self._samples // 2 and len(drawn_b) >= num_texts2:
                    #    drawn_b = []

                pair = SamplePairImpl(pair_class, self.chunk_tokenizer)
                await pair.chunk(chunks_a, chunks_b)
                group_id = PairBuildingProgressEvent.generate_group_id([pair.pair_id])
                await EventBroadcaster.publish("onPairGenerated",
                                               PairBuildingProgressEvent(group_id, pair_num, None,
                                                                         pair, file_names_a, file_names_b),
                                               self.__class__)
                pair_num += 1
                yield pair


class PanParser(CorpusParser):
    """
    Corpus parser for PAN-style authorship verification corpora.

    * `onPairGenerated`: [type PairBuildingProgressEvent]
                         fired when a pair has been generated
    """

    class Class(SamplePairClass):
        UNSPECIFIED = -1
        DIFFERENT_AUTHORS = 0
        SAME_AUTHOR = 1

    async def __aiter__(self) -> AsyncGenerator[SamplePair, None]:
        # parse ground truth if it exists
        ground_truth = {}
        if os.path.isfile(self.corpus_path + "/truth.txt"):
            with open(self.corpus_path + "/truth.txt", "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    tmp = [x.strip().replace("\ufeff", "") for x in re.split("[ \t]+", line)]
                    if 2 != len(tmp):
                        continue
                    ground_truth[tmp[0]] = (tmp[1].upper() == "Y")

        pair_num = 0
        total_num_pairs = len(ground_truth)

        for case_dir in glob(self.corpus_path + "/*"):
            if not os.path.isdir(case_dir) or \
               not os.path.isfile(case_dir + "/unknown.txt") or \
               not os.path.isfile(case_dir + "/known01.txt"):
                continue

            case = os.path.basename(case_dir)

            chunks_a = []
            file_name_a = self.corpus_path + "/" + case + "/unknown.txt"
            chunks_a.append(await self.await_file(file_name_a))

            chunks_b = []
            file_names_b = sorted(glob(self.corpus_path + "/" + case + "/known??.txt"))
            for b in file_names_b:
                chunks_b.append(await self.await_file(b))

            cls = self.Class.UNSPECIFIED
            if case in ground_truth:
                cls = self.Class.SAME_AUTHOR if ground_truth[case] else self.Class.DIFFERENT_AUTHORS

            pair = SamplePairImpl(cls, self.chunk_tokenizer)
            await pair.chunk(chunks_a, chunks_b)
            group_id = PairBuildingProgressEvent.generate_group_id([pair.pair_id])
            await EventBroadcaster.publish("onPairGenerated",
                                           PairBuildingProgressEvent(group_id, pair_num, total_num_pairs,
                                                                     pair, [file_name_a], file_names_b),
                                           self.__class__)

            yield pair
            pair_num += 1
