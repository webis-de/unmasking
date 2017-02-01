from event import EventBroadcaster, ProgressEvent

from input.interfaces import CorpusParser, SamplePair
import os


class BookSampleParser(CorpusParser):
    """
    Parser for book samples. Expects a directory structure where there is one folder
    for each author containing at least two samples of their work.
    Example::

        + Ernest_Hemingway
        |__ + The_Torrents_of_Spring.txt
        |__ + Islands_in_the_Stream.txt
        |__ + The_Garden_of_Eden.txt
        + William_Faulkner
        |__ + Soldier's_Pay.txt
        |__ + Light_in_August.txt

    File and folder names can be chosen arbitrarily, but the book sample files must end in .txt.
    
    Events published by this class:
    
    * `onProgress`: [type: ProgressEvent]
                    fired during author pair generation to indicate current progress
    
    """
    
    class BookSampleParserIterator(CorpusParser.CorpusParserIterator):
        def __init__(self, parser):
            super().__init__(parser)
            
            # file -> author
            self._files = {}
            self._iterator1 = None
            self._next1 = None
            self._current_file_contents = None
            
            # author -> files
            self._authors = {}
            self._iterator2 = None
            self._next2 = None
            
            # de-duplication of single-text pairs
            self._single_text_pairs = []
            
            # read in all directory and file names and build
            # file -> author and author -> files maps
            dirs = os.listdir(self.parser.corpus_path)
            for d in dirs:
                dir_path = os.path.join(self.parser.corpus_path, d)
                if not os.path.isdir(dir_path):
                    continue
                
                files = sorted(os.listdir(dir_path))
                self._authors[d] = []
                for f in files:
                    file_path = os.path.realpath(os.path.join(dir_path, f))
                    if not os.path.isfile(file_path) or not f.endswith(".txt"):
                        continue
                    self._files[file_path] = d
                    self._authors[d].append(file_path)
            
            self._iterator1 = iter(self._files)
            
            # progress publisher
            self._progress_event = ProgressEvent(len(self._files))
        
        def __next__(self) -> SamplePair:
            # next text
            if self._next2 is None:
                EventBroadcaster.publish("onProgress", self._progress_event, self.parser.__class__)
                self._next1 = next(self._iterator1)
                self._iterator2 = iter(self._authors)
                self._progress_event.increment()
                with open(self._next1, "r") as handle:
                    self._current_file_contents = handle.read()
            
            # next author
            try:
                self._next2 = next(self._iterator2)
            except StopIteration:
                self._next2 = None
                return self.__next__()
            
            compare_texts = []
            last_filename = None
            for file_name in self._authors[self._next2]:
                if file_name == self._next1:
                    # don't compare a text with itself
                    continue
                
                with open(file_name, "r") as handle:
                    compare_texts.append(handle.read())
                    last_filename = file_name
            
            num_comp_texts = len(compare_texts)
            if num_comp_texts == 0:
                # if there is only one text of this author, we can't build a pair
                return self.__next__()
            elif num_comp_texts == 1:
                # make sure we don't have the same pair of single texts twice
                pair_set = {self._next1, last_filename}
                if pair_set in self._single_text_pairs:
                    return self.__next__()
                self._single_text_pairs.append(pair_set)
            
            cls = SamplePair.Class.DIFFERENT_AUTHORS
            if self._files[self._next1] == self._next2:
                cls = SamplePair.Class.SAME_AUTHOR
            
            return SamplePair(self._current_file_contents, compare_texts, cls,
                              self.parser.chunk_size, self.parser.language, self.parser.cache_size)
    
    def __iter__(self) -> BookSampleParserIterator:
        return self.BookSampleParserIterator(self)
