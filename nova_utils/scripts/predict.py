"""Standalone script for data predictions

Author:
    Dominik Schiller <dominik.schiller@uni-a.de>
Date:
    06.09.2023

This script performs predictions for annotations in the NOVA database using a provided nova-server module for inference and saves the results to the NOVA-DB.

Args:
    --db_host (str, required): The IP address of the NOVA-DB server.
    --db_port (int, required): The port number of the NOVA-DB server.
    --db_user (str, required): The user to authenticate at the NOVA-DB server.
    --db_password (str, required): The password for the NOVA-DB server user.
    --dataset (str, required): Name of the dataset. Must match entries in NOVA-DB.
    --data_dir (str, required): Path to the NOVA data directory using Windows UNC-Style.
    --sessions (json.loads, required): JSON formatted list of sessions to apply the iterator to.
    --data (json.loads, required): JSON formatted string containing dictionaries that describe the data to load.
    --frame_size (str, optional): Size of the data frame measured in time. Defaults to None.
    --start (str, optional): Start time for processing measured in time. Defaults to None.
    --end (str, optional): End time for processing measured in time. Defaults to None.
    --left_context (str, optional): Left context duration measured in time. Defaults to None.
    --right_context (str, optional): Stride for iterating over data measured in time. If stride is not set explicitly, it will be set to frame_size. Defaults to None.
    --stride (str, optional): Stride for iterating over data measured in time. If stride is not set explicitly, it will be set to frame_size. Defaults to None.
    --add_rest_class (str, optional): Whether to add a rest class for discrete annotations. Defaults to True.
    --fill_missing_data (str, optional): Whether to fill missing data. Defaults to True.
    --cml_dir (str, optional): CML-Base directory for the NOVA-Server module.
    --opt_str (str, optional): JSON formatted string containing dictionaries with key-value pairs, setting the options for a NOVA-Server module.
    --trainer_file_path (str): Path to the trainer file using Windows UNC-Style. (required)

Returns:
    None

Example:
    >>> python predict_annotations.py --dataset "test" --db_host "127.0.0.1" --db_port "37317" --db_user "my_user" --db_password "my_password" --trainer_file_path "test\\test_predict.trainer" --sessions "[\"test_session_1\", \"test_session_2\"]" --data "[{\"src\": \"db:anno\", \"scheme\": \"transcript\", \"annotator\": \"test\", \"role\": \"testrole\"}]" --frame_size "0" --left_context "0" --right_context "0" --job_i_d "test_job" --opt_str "num_speakers=2;speaker_ids=testrole,testrole2" --cml_dir "./cml" --data_dir "./data" --log_dir "./log" --cache_dir "./cache" --tmp_dir "./tmp"
"""

import argparse
from pathlib import Path, PureWindowsPath
from nova_utils.utils import ssi_xml_utils
from nova_utils.data.provider.nova_iterator import NovaIterator
from nova_utils.scripts.parsers import (
    nova_db_parser,
    nova_iterator_parser,
    nova_server_module_parser,
)
from nova_utils.data.handler.mongo_handler import AnnotationHandler
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
    # print(args)

    # Create argument groups
    db_args, _ = nova_db_parser.parse_known_args()
    iter_args, _ = nova_iterator_parser.parse_known_args()
    module_args, _ = nova_server_module_parser.parse_known_args()

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

    # Load Trainer
    model_script_path = (
        trainer_file_path.parent / PureWindowsPath(trainer.model_script_path)
    ).resolve()
    source = SourceFileLoader(
        "ns_cl_" + model_script_path.stem, str(model_script_path)
    ).load_module()
    print(f"Trainer module {Path(model_script_path).name} loaded")

    trainer_class = getattr(source, trainer.model_create)
    predictor = trainer_class()
    print(f"Model {trainer.model_create} created")

    # TODO implement SSI interface
    # If the module implements the Trainer interface load weights
    # if isinstance(predictor, iTrainer):
    #     # Load Model
    #     model_weight_path = (
    #             trainer_file_path.parent / trainer.model_weights_path
    #     )
    #     logger.info(f"Loading weights from {model_weight_path}")
    #     predictor.load(model_weight_path)
    #     logger.info("Model loaded.")

    # Init database handler
    db_handler = AnnotationHandler(**vars(db_args))

    # Iterate over all sessions
    for ds_iter in iterators:
        print(f"Predict session {ds_iter.sessions[0]}...")
        try:
            data = predictor.process_data(ds_iter)
            annos = predictor.to_anno(data)
        except FileNotFoundError as e:
            print(
                f"\tIterator exited with error: '{str(e)}'. Continuing with next session."
            )
            caught_ex = True
            continue
        finally:
            print("...done")

        print("Saving predictions to database...")

        try:
            for anno in annos:
                db_handler.save(anno)
        except FileExistsError as e:
            print(
                f"\tCould note save annotation: '{str(e)}'. Continuing with next session."
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
