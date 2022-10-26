# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser, Namespace
from functools import partial
import html
import json
import os
import re
from typing import Dict, Tuple
from multiprocessing import Pool
from tqdm import tqdm
from lxml import etree


def extract_text(input_text: str, keep_markup: bool = False):
    r""" Clean text from markup and other junk. """
    input_text = input_text.replace("\n", "").replace("\r", "")

    if keep_markup:
        text = html.unescape(input_text)

    elif not keep_markup:
        text_root = etree.HTML(input_text)

        if text_root is None:
            return None

        text = " ".join(text_root.itertext())
        text = re.sub(" +", " ", text)
        text = text.encode("ascii", "xmlcharrefreplace").decode("utf-8")
        text = html.unescape(text)

    return text


def generate_closed_book_format(paths: Tuple[str], only_english: bool = True, keep_markup: bool = False) -> Dict:
    r""" Process a single json file and output dictionary of question and answers. """
    input_path, output_path = paths

    with open(input_path, "r") as fi:
        output = dict()

        for website in fi:
            content = json.loads(website)

            if only_english and content["Fasttext_language"] != "en":
                continue

            questions = content["Questions"]

            for question in questions:
                question_text = ""
                answer_list = []

                if "name_markup" in question.keys():
                    extracted_text = extract_text(question["name_markup"], keep_markup=keep_markup)
                    if extracted_text is not None:
                        question_text += extracted_text + " "

                if "text_markup" in question.keys():
                    extracted_text = extract_text(question["text_markup"], keep_markup=keep_markup)
                    if extracted_text is not None:
                        question_text += extracted_text

                if len(question_text) > 0:
                    for answer in question["Answers"]:
                        if "text_markup" in answer.keys():
                            answer_text = extract_text(
                                answer["text_markup"], keep_markup=keep_markup
                            )
                        else:
                            answer_text = None

                        if answer_text:
                            answer_list.append(answer_text)

                if question_text and answer_list:
                    output[question_text] = answer_list

    with open(output_path, "w") as fo:
        json.dump(output, fo)


def main(args: Namespace):

    assert os.path.isdir(args.input_folder)
    assert not os.path.isdir(args.output_folder)

    os.makedirs(args.output_folder, exist_ok=False)

    files = []
    for filename in os.listdir(args.data_folder):
        files.append((
            os.path.join(args.input_folder, filename),
            os.path.join(args.output_folder, filename),
        ))

    pool = Pool(args.num_workers)
    fn_worker = partial(generate_closed_book_format, only_english=args.only_english, keep_markup=args.keep_markup)
    
    tqdm(pool.map(fn_worker, files), desc="Processing...", total=len(files))


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Generate sequence-to-sequence input and output for closed-book QA"
    )
    parser.add_argument("--input_folder", help="Path to the json dataset")
    parser.add_argument("--output_folder", help="Path to the output file")
    parser.add_argument(
        "--only_english",
        action="store_true",
        help="Only keep english samples in the dataset",
    )
    parser.add_argument(
        "--keep_markup", action="store_true", help="Keep the HTML markup"
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Use multiprocessing"
    )
    main(parser.parse_args())
