"""Standalone script for data extraction

Author:
    Dominik Schiller <dominik.schiller@uni-a.de>
Date:
    06.09.2023

This script utilizes novaserver modules to extract datastreams from either NOVA-DB annotations and datastreams.

.. argparse::
   :module: nova_utils.scripts.predict
   :func: parser
   :prog: nu-predict

Returns:
    None

Example:
    >>> nu-extract --dataset "test" --db_host "127.0.0.1" --db_port "37317" --db_user "my_user" --db_password "my_password" --chain_file_path "test\\test_extract.chain" --sessions "[\"test_session_1\", \"test_session_2\"]" --data "[{\"src\": \"db:anno\", \"scheme\": \"transcript\", \"annotator\": \"test\", \"role\": \"testrole\"}]" --frame_size "0" --left_context "0" --right_context "0" --job_i_d "test_job" --opt_str "num_speakers=2;speaker_ids=testrole,testrole2" --cml_dir "./cml" --data_dir "./data" --log_dir "./log" --cache_dir "./cache" --tmp_dir "./tmp"
"""

import argparse
from pathlib import Path, PureWindowsPath
from nova_utils.utils.ssi_xml_utils import Chain, ChainLink
from nova_utils.data.provider.nova_iterator import NovaIterator
from nova_utils.scripts.parsers import (
    nova_db_parser,
    nova_iterator_parser,
    nova_server_module_parser,
)
from nova_utils.data.handler.mongo_handler import StreamHandler
from importlib.machinery import SourceFileLoader

# Main parser for predict specific options
parser = argparse.ArgumentParser(
    description="Use a provided nova-server module for inference and save results to NOVA-DB",
    parents=[nova_db_parser, nova_iterator_parser, nova_server_module_parser],
)
parser.add_argument(
    "--chain_file_path",
    type=str,
    required=True,
    help="Path to the chain file using Windows UNC-Style",
)

def _main():

    args, _ = parser.parse_known_args()

    # Create argument groups
    db_args, _ = nova_db_parser.parse_known_args()
    iter_args, _ = nova_iterator_parser.parse_known_args()
    module_args, _ = nova_server_module_parser.parse_known_args()

    caught_ex = False

    # Load chain
    chain = Chain()
    chain_file_path = Path(module_args.cml_dir).joinpath(
        PureWindowsPath(args.chain_file_path)
    )
    if not chain_file_path.is_file():
        raise FileNotFoundError(f"Chain file not available: {chain_file_path}")
    else:
        chain.load_from_file(chain_file_path)
        print("Chain successfully loaded.")


    # Build data loaders
    sessions = iter_args.sessions
    iterators = []

    # TODO split for role if multirole input is false
    args = {**vars(db_args), **vars(iter_args)}

    for session in sessions:
        print(session)
        args["sessions"] = [session]
        ni = NovaIterator(**args)
        iterators.append(ni)
    print("Data iterators initialized")

    # Init database handler
    stream_handler = StreamHandler(**vars(db_args), data_dir=iter_args.data_dir)

    # Iterate over all sessions
    for ds_iter in iterators:
        print(f"Extract data for session {ds_iter.sessions[0]}...")

        # Iterate over all chain links
        cl: ChainLink
        for i, cl in enumerate(chain.links):
            model_script_path = chain_file_path.parent / cl.script
            source = SourceFileLoader(
                "ns_cl_" + model_script_path.stem, str(model_script_path)
            ).load_module()
            extractor_class = getattr(source, cl.create)
            extractor = extractor_class()
            print(f"Model {cl.create} created")

            # Check if there will be more chain links to execute
            if i + 1 != len(chain.links):
                raise NotImplementedError('Chains with more than one element will be supported in future releases.')
                # Assert chainability
                # if not extractor.chainable:
                #     raise AssertionError(
                #         "Extraction module does not not support further chaining but is not the last link in chain."
                #     )
                # # TODO implement nova data types in the server module
                # # TODO process data and write tmp session files to disk if necessary
                # print("Extract data...")
                #
                # try:
                #     data = extractor.process_data(ds_iter)
                #     ds_iter = extractor.to_ds_iterable(data)
                # except Exception as e:
                #     print(
                #         f"\tIterator exited with error: '{str(e)}'. Continuing with next session."
                #     )
                #     caught_ex = True
                #     continue
                #
                # print("...done")

            # Last element of chain
            else:
                try:
                    data = extractor.process_data(ds_iter)
                    streams = extractor.to_stream(data)
                except FileNotFoundError as e:
                    print(
                        f"\tIterator exited with error: '{str(e)}'. Continuing with next session."
                    )
                    caught_ex = True
                    break

                print("...done")

                print("Saving streams to disk...")
                try:
                    for stream in streams:
                        print(f"\t{stream.meta_data.name}")
                        stream_handler.save(stream)
                except FileExistsError as e:
                    print(
                        f"\tCould note save stream: '{str(e)}'. Continuing with next session."
                    )
                    caught_ex = True
                    continue
                finally:
                    print("...done")

    print("Prediction completed!")
    if caught_ex:
        print(
            "Prediction job encountered errors for some sessions. Check logs for details."
        )
        exit(1)


if __name__ == "__main__":
    _main()
 