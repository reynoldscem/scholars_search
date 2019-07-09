'''Script for obtaining stats about authors from google scholar'''
from argparse import ArgumentParser
from scholarly import search_author
from joblib import Parallel, delayed
import os


MAX_AUTHORS = 3
N_JOBS = 4

KEYWORDS = [
    'image registration', 'deformable', 'non-rigid', 'shape matching',
    'convolutional', 'cnn', 'neural', 'medical image',
    'learning', 'registration', 'supervised', 'deep',
    'adversarial', 'affine',
    'rigid', 'diffeomorphism', 'spline',
    'deformation', 'appearance', 'elastic',
    'alignment'
]
# 'loss', 'network', 'information', 'models',


def build_parser():
    parser = ArgumentParser(description=__doc__)

    parser.add_argument(
        'author_filename',
        help='Path to text file containing list of authors'
    )

    parser.add_argument(
        '-j', '--n-jobs',
        default=N_JOBS,
        help='Number of parallel jobs to use.'
    )

    return parser


def assert_file_exists(filename):
    message = '{} does not exist.'.format(filename)
    assert os.path.isfile(filename), message


def load_authors(filename):
    with open(filename) as fd:
        author_list = fd.read().splitlines()

    author_list = list(sorted(set(author_list)))

    return author_list


def rank_authors(authors):
    from collections import defaultdict
    from itertools import islice

    author_to_match_count = defaultdict(int)

    authors = list(islice(authors, MAX_AUTHORS))

    if len(authors) == 0:
        raise StopIteration
    elif len(authors) == 1:
        return authors

    for index, author in enumerate(authors):
        # print('{} {}'.format(index, author.name))
        publications = author.fill().publications
        print('{} publications:'.format(len(publications)))
        for publication in publications:
            title = publication.bib['title'].lower()

            count_for_pub = 0
            for keyword in KEYWORDS:
                count_for_pub += title.count(keyword)
            # if count_for_pub > 0:
            #     print(title)
            #     print(count_for_pub)
            author_to_match_count[author] += count_for_pub
        # keyword_matches = author_to_match_count[author]
        # print('{} keyword matches'.format(keyword_matches), flush=True)
        # print()

    return list(sorted(
        author_to_match_count,
        key=author_to_match_count.get,
        reverse=True
    ))


def get_author_info(author_name, callback=None):
    try:
        query = search_author(author_name)
        potential_authors = list(query)
        best_author = rank_authors(potential_authors)[0]
        author_result = best_author.fill()
        if callback:
            callback(author_name, author_result)
    except StopIteration:
        print('{} not found!'.format(author_name))

        return
    except Exception as e:
        print(
            'Exception occurred for {}! Message: {}'.format(author_name, e),
            flush=True
        )

        return

    return author_result


def print_author_result(author_name, author_result):
    print(
        '{} ({}, {}) cited by {}, h-index of {}'
        ''.format(
            author_result.name, author_name, author_result.id,
            author_result.citedby, author_result.hindex
        ),
        flush=True
    )


def main():
    args = build_parser().parse_args()
    assert_file_exists(args.author_filename)

    authors = load_authors(args.author_filename)

    print('{} total authors'.format(len(authors)))
    print()

    author_infos = Parallel(n_jobs=args.n_jobs)(
        delayed(get_author_info)(author, print_author_result)
        for author in authors
    )


if __name__ == '__main__':
    main()
