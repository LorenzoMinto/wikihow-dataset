import sys
import os
import hashlib
import subprocess
import collections

import json
import tarfile
import io
import pickle as pkl
import pandas as pd
import spacy

import nltk

dm_single_close_quote = '\u2019' # unicode
dm_double_close_quote = '\u201d'
# acceptable ways to end a sentence
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"',
              dm_single_close_quote, dm_double_close_quote, ")"]

wikihow_articles_tokenized = "wikihow_articles_tokenized"

"""Add here incrementally new conditioning hypothesis, increment the final subscript"""
def conditioning_hyp1(df):
    return df['title'] + " $ " + df['text']

def tokenize_articles(articles_csv, tokenized_stories_csv):
    """Maps a whole csv of articles to its tokenized version using
       Stanford CoreNLP Tokenizer
    """
    print("Preparing to tokenize {} to {}...".format(articles_csv,
                                                     tokenized_stories_csv))

    wikihow_df = pd.read_csv(articles_csv)
    wikihow_df = wikihow_df[wikihow_df['text'].isna() == False]
    # Change conditioning substructure here by redefining or adding conditioning function
    wikihow_df['conditioning'] = conditioning_hyp1(wikihow_df)
    wikihow_df_c1 = wikihow_df.drop(['title', 'text'], axis=1)
    wikihow_df_c1.rename(columns={'headline': 'summary'}, inplace=True)

    nltk.download('punkt')

    #Remove empty lines
    wikihow_df_c1 = wikihow_df_c1.applymap(lambda x: [" ".join(nltk.word_tokenize(line)) for line in x.split('\n') if line != ''])

    wikihow_df_c1.to_csv(tokenized_stories_csv)
    print("Successfully finished tokenizing {} to {}.\n".format(
        articles_csv, tokenized_stories_csv))


def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if "@highlight" in line:
        return line
    if line == "":
        return line
    if line[-1] in END_TOKENS:
        return line
    return line + " ."


def write_to_tar(out_file, random=True, makevocab=False):
    """Reads the tokenized articles, splits them in train,test,val and writes them to a out_file.
    """

    wikihow_df = pd.read_csv(wikihow_articles_tokenized)
    articles = wikihow_df['conditioning'].to_numpy()
    summaries = wikihow_df['summary'].to_numpy()

    if makevocab:
        vocab_counter = collections.Counter()

    with tarfile.open(out_file+".tar", 'w') as writer:
        articles_n = articles.size
        for idx in range(articles_n):
            if idx % 1000 == 0:
                print("Writing story {} of {}; {:.2f} percent done".format(
                    idx, articles_n, float(idx)*100.0/float(articles_n)))

            # Get the strings to write to .bin file
            article_sents = articles[idx]
            summary_sents = summaries[idx]

            # Write to JSON file
            js_example = {}
            js_example['id'] = "{}".format(idx)
            js_example['article'] = article_sents
            js_example['summary'] = summary_sents
            js_serialized = json.dumps(js_example, indent=4).encode()
            save_file = io.BytesIO(js_serialized)
            tar_info = tarfile.TarInfo('{}/{}.json'.format(
                os.path.basename(out_file).replace('.tar', ''), idx))
            tar_info.size = len(js_serialized)
            writer.addfile(tar_info, save_file)

            # Write the vocab to file, if applicable
            if makevocab:
                art_tokens = ' '.join(article_sents).split()
                sum_tokens = ' '.join(summary_sents).split()
                tokens = art_tokens + sum_tokens
                tokens = [t.strip() for t in tokens] # strip
                tokens = [t for t in tokens if t != ""] # remove empty
                vocab_counter.update(tokens)

    print("Finished writing file {}\n".format(out_file))

    # write vocab to file
    if makevocab:
        print("Writing vocab file...")
        with open("vocab_cnt.pkl",
                  'wb') as vocab_file:
            pkl.dump(vocab_counter, vocab_file)
        print("Finished writing vocab file")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("USAGE: python make_wh_datafiles.py"
              " <wikihow_articles_csv>")
        sys.exit()
    wh_articles_csv = sys.argv[1]

    # Run stanford tokenizer on both stories dirs,
    # outputting to tokenized stories directories
    tokenize_articles(wh_articles_csv, wikihow_articles_tokenized)

    # Read the tokenized stories, do a little postprocessing
    # then write to bin files
    write_to_tar(wikihow_articles_tokenized, random=True, makevocab=True)
