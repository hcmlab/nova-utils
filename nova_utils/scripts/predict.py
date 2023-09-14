"""General logic for predicting annotations to the nova database
Author: Dominik Schiller <dominik.schiller@uni-a.de>
Date: 06.09.2023
"""
import argparse
from pathlib import Path, PureWindowsPath
from nova_utils.utils import ssi_xml_utils
from nova_utils.data.provider.nova_iterator import NovaIterator
from nova_utils.scripts.parsers import nova_db_parser, nova_iterator_parser, nova_server_module_parser
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

def main():
    # Parse arguments
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

if __name__ == '__main__':
    main()