"""Standalone script for general processing

Author:
    Dominik Schiller <dominik.schiller@uni-a.de>
Date:
    20.09.2023

This script performs generall data processing to extract either annotations to NOVA-Database or streams to disk using a provided nova-server module for inference.

.. argparse::
   :module: nova_utils.scripts.process
   :func: parser
   :prog: nu-process

Returns:
    None

Example:
    >>> nu-prcess --dataset "test" --db_host "127.0.0.1" --db_port "37317" --db_user "my_user" --db_password "my_password" --trainer_file_path "test\\test_predict.trainer" --sessions "[\"test_session_1\", \"test_session_2\"]" --data "[{\"src\": \"db:anno\", \"scheme\": \"transcript\", \"annotator\": \"test\", \"role\": \"testrole\"}]" --frame_size "0" --left_context "0" --right_context "0" --job_i_d "test_job" --opt_str "num_speakers=2;speaker_ids=testrole,testrole2" --cml_dir "./cml" --data_dir "./data" --log_dir "./log" --cache_dir "./cache" --tmp_dir "./tmp"
"""

import argparse
import os
from typing import Union, Type
from pathlib import Path, PureWindowsPath
from nova_utils.utils import ssi_xml_utils, string_utils
from nova_utils.data.provider.nova_iterator import NovaIterator
from nova_utils.scripts.parsers import (
    nova_db_parser,
    nova_iterator_parser,
    nova_server_module_parser,
)
from nova_utils.interfaces.server_module import Predictor, Extractor
from nova_utils.data.handler import mongo_handler as db_handler
from importlib.machinery import SourceFileLoader

# Main parser for predict specific options
parser = argparse.ArgumentParser(
    description="Use a provided nova-server module for inference and save results to NOVA-DB",
    parents=[nova_db_parser, nova_iterator_parser, nova_server_module_parser],
)
parser.add_argument(
    "--trainer_file_path",
    type=str,
    required=True,
    help="Path to the trainer file using Windows UNC-Style",
)


def _main():

    args, _ = parser.parse_known_args()

    # Create argument groups
    db_args, _ = nova_db_parser.parse_known_args()
    iter_args, _ = nova_iterator_parser.parse_known_args()
    module_args, _ = nova_server_module_parser.parse_known_args()

    # Set environment variables
    os.environ['CACHE_DIR'] = module_args.cache_dir
    os.environ['TMP_DIR'] = module_args.tmp_dir

    caught_ex = False

    # Load trainer
    trainer = ssi_xml_utils.Trainer()
    trainer_file_path = Path(module_args.cml_dir).joinpath(
        PureWindowsPath(args.trainer_file_path)
    )
    if not trainer_file_path.is_file():
        raise FileNotFoundError(f"Trainer file not available: {trainer_file_path}")
    else:
        trainer.load_from_file(trainer_file_path)
        print("Trainer successfully loaded.")

    # Load module
    if not trainer.model_script_path:
        raise ValueError('Trainer has no attribute "script" in model tag.')

    model_script_path = (
        trainer_file_path.parent / PureWindowsPath(trainer.model_script_path)
    ).resolve()
    source = SourceFileLoader(
        "ns_cl_" + model_script_path.stem, str(model_script_path)
    ).load_module()
    print(f"Trainer module {Path(model_script_path).name} loaded")
    opts = string_utils.parse_nova_option_string(args.opt_str)
    processor_class: Union[Type[Predictor], Type[Extractor]] = getattr(
        source, trainer.model_create
    )
    processor = processor_class(model_io=trainer.meta_io, opts=opts, trainer=trainer)
    print(f"Model {trainer.model_create} created")

    # Build data loaders
    sessions = iter_args.sessions
    iterators = []
    args = {**vars(db_args), **vars(iter_args)}

    for session in sessions:
        print(session)
        args["sessions"] = [session]
        ni = NovaIterator(**args)
        iterators.append(ni)
    print("Data iterators initialized")

    # Init data handler
    annotation_handler = db_handler.AnnotationHandler(**vars(db_args))
    stream_handler = db_handler.StreamHandler(**vars(db_args), data_dir=iter_args.data_dir)

    # Iterate over all sessions
    for ds_iter in iterators:
        annos = []
        streams = []

        # Data processing
        print(f"Process session {ds_iter.sessions[0]}...")
        try:
            data = processor.process_data(ds_iter)

            if isinstance(processor, Predictor):
                annos = processor.to_anno(data)

            if isinstance(processor, Extractor):
                streams = processor.to_stream(data)

        except FileNotFoundError as e:
            print(
                f"\tProcessor exited with error: '{str(e)}'. Continuing with next session."
            )
            caught_ex = True
            continue
        finally:
            print("...done")

        # Data Saving
        if annos:
            print("Saving annotations to database...")
            for anno in annos:
                try:
                    annotation_handler.save(anno, overwrite=True)
                except FileExistsError as e:
                    print(f"\tCould not save annotation: '{str(e)}' ")
                    caught_ex = True
            print("...done")

        if streams:
            print("Saving streams to disk...")
            for stream in streams:
                try:
                    stream_handler.save(stream)
                except FileExistsError as e:
                    print(f"\tCould not save stream: '{str(e)}'")
                    caught_ex = True
            print("...done")

    print("Processing completed!")
    if caught_ex:
        print(
            "Processing job encountered errors for some sessions. Check logs for details."
        )
        exit(1)


if __name__ == "__main__":
    _main()
