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
    >>> nu-process --dataset "test" --db_host "127.0.0.1" --db_port "37317" --db_user "my_user" --db_password "my_password" --trainer_file_path "test\\test_predict.trainer" --sessions "[\"test_session_1\", \"test_session_2\"]" --data "[{\"src\": \"db:anno\", \"scheme\": \"transcript\", \"annotator\": \"test\", \"role\": \"testrole\"}]" --frame_size "0" --left_context "0" --right_context "0" --job_i_d "test_job" --opt_str "num_speakers=2;speaker_ids=testrole,testrole2" --cml_dir "./cml" --data_dir "./data" --log_dir "./log" --cache_dir "./cache" --tmp_dir "./tmp"
"""

import argparse
import sys
import os
import shutil
from importlib.machinery import SourceFileLoader
from pathlib import Path, PureWindowsPath
from typing import Union, Type

from nova_utils.data.annotation import Annotation
from nova_utils.data.handler import nova_db_handler as db_handler
from nova_utils.data.provider.data_manager import DatasetManager, SessionManager
from nova_utils.data.provider.nova_dataset_iterator import NovaDatasetIterator
from nova_utils.data.stream import Stream
from nova_utils.interfaces.server_module import Predictor, Extractor
from nova_utils.scripts.parsers import (
    dm_parser,
    nova_db_parser,
    request_parser,
    nova_iterator_parser,
    nova_server_module_parser,
)
from nova_utils.utils import ssi_xml_utils, string_utils
from nova_utils.utils.string_utils import string_to_bool

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


def main(args):

    process_args, _ = parser.parse_known_args(args)

    # Create argument groups
    db_args, _ = nova_db_parser.parse_known_args(args)
    req_args, _ = request_parser.parse_known_args(args)
    dm_args, _ = dm_parser.parse_known_args(args)
    iter_args, _ = nova_iterator_parser.parse_known_args(args)
    module_args, _ = nova_server_module_parser.parse_known_args(args)

    # Set environment variables
    os.environ['CACHE_DIR'] = module_args.cache_dir
    os.environ['TMP_DIR'] = module_args.tmp_dir

    caught_ex = False

    # Load trainer
    trainer = ssi_xml_utils.Trainer()
    trainer_file_path = Path(module_args.cml_dir).joinpath(
        PureWindowsPath(process_args.trainer_file_path)
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
    opts = string_utils.parse_nova_option_string(process_args.opt_str)
    processor_class: Union[Type[Predictor], Type[Extractor]] = getattr(
        source, trainer.model_create
    )
    processor = processor_class(model_io=trainer.meta_io, opts=opts, trainer=trainer)
    print(f"Model {trainer.model_create} created")

    # Build data loaders
    #args = {**vars(db_args), **vars(dm_args)}

    ctx = {
        'db' : {
            **vars(db_args)
        },
        'request' : {
            **vars(req_args)
        }
    }

    # Clear output for job id
    shared_dir = ctx['request'].get('shared_dir')
    job_id = ctx['request'].get('job_id')
    if shared_dir and job_id:
        output_dir = Path(shared_dir) / job_id
        if output_dir.exists():
            if output_dir.is_dir():
                shutil.rmtree(output_dir)
        if output_dir.is_file():
            output_dir.unlink()

    single_session_datasets = []
    is_iterable = string_to_bool(trainer.meta_is_iterable)

    for session in dm_args.sessions:
        if is_iterable:
            dataset_manager = NovaDatasetIterator(dataset=dm_args.dataset, data_description=dm_args.data, source_context=ctx, session_names=[session], **vars(iter_args))
        else:
            dataset_manager = DatasetManager(dataset=dm_args.dataset, data_description=dm_args.data, source_context=ctx, session_names=[session])

        single_session_datasets.append(dataset_manager)

    # iterators = []
    # sessions = iter_args.sessions
    # for session in sessions:
    #     print(session)
    #     args["sessions"] = [session]
    #     ni = NovaIterator(**args)
    #     iterators.append(ni)
    print("Data managers initialized")

    # Iterate over all sessions
    for ss_dataset in single_session_datasets:
        session = ss_dataset.session_names[0]

        if not is_iterable:
            ss_dataset.load()

        # Data processing
        print(f"Process session {session}...")
        try:
            # Todo add iterator option
            data_processed = processor.process_data(dataset_manager)
            data_output = processor.to_output(data_processed)
        except FileNotFoundError as e:
            print(
                f"\tProcessor exited with error: '{str(e)}'. Continuing with next session."
            )
            caught_ex = True
            continue
        finally:
            print("...done")

        # TODO deprecation warning
        if isinstance(data_output, list):
            # Init data handler
            annotation_handler = db_handler.AnnotationHandler(**vars(db_args))
            stream_handler = db_handler.StreamHandler(**vars(db_args), data_dir=process_args.data_dir)
            for out in data_output:

                if isinstance(out, Annotation):
                    print("Saving annotation to database...")
                    try:
                        annotation_handler.save(out, overwrite=True)
                    except FileExistsError as e:
                        print(f"\tCould not save annotation: '{str(e)}' ")
                        caught_ex = True
                    print("...done")

                elif isinstance(out, Stream):
                    print("Saving stream to disk...")
                    try:
                        stream_handler.save(out)
                    except FileExistsError as e:
                        print(f"\tCould not save stream: '{str(e)}'")
                        caught_ex = True
                    print("...done")
        else:
            # Data Saving
            if isinstance(data_output, dict):
                session_manager : SessionManager
                session_manager = ss_dataset.sessions[session]['manager']
                for io_id, data_object in data_output.items():
                    session_manager.output_data_templates[io_id] = data_object

            ss_dataset.save()

    print("Processing completed!")
    if caught_ex:
        print(
            "Processing job encountered errors for some sessions. Check logs for details."
        )
        exit(1)

# Entry point for nu-process
def cl_main():
    main(sys.argv[1:])

# Entry point for python
if __name__ == "__main__":
    main(sys.argv[1:])
