#!/usr/bin/env python3

from input import BookSampleParser
from classifier import UniqueRandomUndersampler


def main():
    parser = BookSampleParser("corpora", 500, "english")
    s = UniqueRandomUndersampler()
    for p in parser:
        print(len(list(s.generate_chunk_pairs(p))))

if __name__ == "__main__":
    main()
