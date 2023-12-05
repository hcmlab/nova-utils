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
import json
import numpy as np

from nova_utils.data.annotation import Annotation
from nova_utils.data.handler import nova_db_handler as db_handler
from nova_utils.data.provider.data_manager import DatasetManager, SessionManager
from nova_utils.data.provider.data_manager import DatasetManager, SessionManager
from nova_utils.data.provider.dataset_iterator import DatasetIterator
from nova_utils.data.stream import Stream
from nova_utils.interfaces.server_module import Predictor, Extractor
from nova_utils.scripts.parsers import (
    dm_parser,
    nova_db_parser,
    request_parser,
    nova_iterator_parser,
    nova_server_module_parser,
)
from nova_utils.utils import ssi_xml_utils, string_utils, request_utils
from nova_utils.utils.string_utils import string_to_bool
from nova_utils.explainer.dice import dice_explain
from nova_utils.explainer.lime_explainer import lime_image
from nova_utils.explainer.lime_explainer import lime_tabular
from nova_utils.explainer.tf_explainer import tf_explainer
from nova_utils.explainer.cycle_gan_explainer import style_conversion

from nova_utils.data.static import Image, Text

# Main parser for predict specific options
parser = argparse.ArgumentParser(
    description="Use a provided nova-server module for inference and save results to NOVA-DB",
    parents=[nova_db_parser, nova_iterator_parser, nova_server_module_parser],
)

parser.add_argument(
    "--explainer_type",
    type=str,
    required=True,
    help="Explainer to be used",
)

parser.add_argument(
    "--trainer_file_path",
    type=str,
    required=True,
    help="Path to the trainer file using Windows UNC-Style",
)

parser.add_argument(
    "--explainer_file_path",
    type=str,
    required=False,
    help="Path to the explainer file using Windows UNC-Style",
)

parser.add_argument(
    "--frame_id",
    type=int,
    required=True,
    help="frame to be explained",
)

parser.add_argument(
    "--class_counterfactual",
    type=int,
    required=False,
    help="counterfactual class",
)

parser.add_argument(
    "--num_counterfactuals",
    type=int,
    required=False,
    help="number of counterfactuals to be created",
)

parser.add_argument(
    "--num_features",
    type=int,
    required=False,
    help="LIME: number of features",
)

parser.add_argument(
    "--top_labels",
    type=int,
    required=False,
    help="LIME: top labels",
)

parser.add_argument(
    "--top_class",
    type=int,
    required=False,
    help="LIME: top class",
)

parser.add_argument(
    "--num_samples",
    type=int,
    required=False,
    help="LIME: number of samples",
)

parser.add_argument(
    "--hide_color",
    type=bool,
    required=False,
    help="LIME: hide color",
)

parser.add_argument(
    "--hide_rest",
    type=bool,
    required=False,
    help="LIME: hide rest",
)

parser.add_argument(
    "--positive_only",
    type=bool,
    required=False,
    help="LIME: positive only",
)

parser.add_argument(
    "--tf_explainer",
    type=str,
    required=False,
    help="Tf_explain: Explainer",
)

def main(args):

    explainer_args, _ = parser.parse_known_args(args)

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

    explainer = ssi_xml_utils.Explainer()
    if not explainer_args.explainer_file_path is None:

        # Load explainer
        explainer_file_path = Path(module_args.cml_dir).joinpath(
            PureWindowsPath(explainer_args.explainer_file_path)
        )
        
        if not explainer_file_path.is_file():
            raise FileNotFoundError(f"Explainer file not available: {explainer_file_path}")
        else:
            explainer.load_from_file(explainer_file_path)
            print("Explainer successfully loaded.")

        # Load module
        if not explainer.model_script_path:
            raise ValueError('Explainer has no attribute "script" in model tag.')

        model_script_path = (
                explainer_file_path.parent / PureWindowsPath(explainer.model_script_path)
        ).resolve()
        source = SourceFileLoader(
            "ns_cl_" + model_script_path.stem, str(model_script_path)
        ).load_module()
        print(f"Explainer module {Path(model_script_path).name} loaded")
        opts = module_args.options
        if module_args.options is None:
            opts = string_utils.parse_nova_option_string(explainer_args.opt_str)
            print('Option --opt_str is deprecated. Use --options in the future.')

        opts["explainer_module_path"] = explainer_file_path.parent
        explainer_class: Union[Type[Predictor], Type[Extractor]] = getattr(
            source, explainer.model_create
        )

    # Load trainer
    trainer = ssi_xml_utils.Trainer()
    trainer_file_path = Path(module_args.cml_dir).joinpath(
        PureWindowsPath(explainer_args.trainer_file_path)
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
    opts = module_args.options
    if module_args.options is None:
        opts = string_utils.parse_nova_option_string(explainer_args.opt_str)
        print('Option --opt_str is deprecated. Use --options in the future.')

    opts["trainer_module_path"] = trainer_file_path.parent
    processor_class: Union[Type[Predictor], Type[Extractor]] = getattr(
        source, trainer.model_create
    )

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

    #single_session_datasets = []
    single_session_data_provider = []
    is_iterable = string_to_bool(trainer.meta_is_iterable)

    for session in dm_args.sessions:
       if is_iterable:
           dataset_manager = DatasetIterator(dataset=dm_args.dataset, data_description=dm_args.data, source_context=ctx, session_names=[session], **vars(iter_args))
       else:
           dataset_manager = DatasetManager(dataset=dm_args.dataset, data_description=dm_args.data, source_context=ctx, session_names=[session])#

       single_session_data_provider.append(dataset_manager)

    # for session in dm_args.sessions:
    #     requires_db = any([request_utils.parse_src_tag(dd)[0] == request_utils.Origin.DB.value for dd in dm_args.data])
    #     data_provider_cls = NovaDatasetManager if requires_db else DatasetManager
    #     data_manager = data_provider_cls(dataset=dm_args.dataset, data_description=dm_args.data, source_context=ctx, session_names=[session])
    #     single_session_data_manager.append(data_manager)
    print("Data managers initialized")

    # Iterate over all sessions
    #for ss_dataset in single_session_datasets:
    #    session = ss_dataset.session_names[0]#

    #    if not is_iterable:
    #        ss_dataset.load()
    # Iterate over all sessions
    for provider in single_session_data_provider:
        session = provider.session_names[0]
        #data_provider = provider
        sm = provider.sessions[session]["manager"]

        try:
            if isinstance(provider, DatasetManager):
                provider.load()

            # Data processing
            print(f"Process session {session}...")
            single_frame = sm.input_data["explanation_stream"].data[explainer_args.frame_id]

            if is_iterable:
                data_provider = DatasetIterator(provider, **vars(iter_args))
                stream_data = []
                anno_data = []

                for v in data_provider:
                    if v["explanation_stream"].shape[0] == 0:
                        continue
                    stream_data.append(v["explanation_stream"][0])
                    anno_data.append(v["explanation_anno"])


            # if ml backend is tensorflow postpone model initialization after creating data streams as tensorflow allocates all available memory
            if explainer_args.explainer_type == "CYCLE_GAN":
                explainer = explainer_class(model_io=trainer.meta_io, opts=opts, trainer=explainer)
                data_output = style_conversion(single_frame, explainer.get_explainable_model())
            else:
                processor = processor_class(model_io=trainer.meta_io, opts=opts, trainer=trainer)
                print(f"Model {trainer.model_create} created")


                # Todo add iterator option
                model = processor.get_explainable_model()
                expl_func = processor.get_predict_function()

                
                if explainer_args.explainer_type == "LIME_IMAGE":
                    data_output = lime_image(single_frame, explainer_args.num_features, explainer_args.top_labels, explainer_args.num_samples, explainer_args.hide_color, explainer_args.hide_rest, explainer_args.positive_only, model)
                elif explainer_args.explainer_type == "LIME_TABULAR":
                    data_output = lime_tabular(stream_data, explainer_args.frame_id, explainer_args.top_class, explainer_args.num_features, model)
                elif explainer_args.explainer_type == "DICE":
                    data_output = dice_explain(stream_data, anno_data, explainer_args.frame_id, sm.input_data["explanation_stream"].meta_data.sample_shape[0], sm.input_data["explanation_anno"].annotation_scheme, trainer.meta_backend, explainer_args.class_counterfactual, explainer_args.num_counterfactuals, model)
                elif explainer_args.explainer_type == "TF_EXPLAIN":
                    data_output = tf_explainer(single_frame, explainer_args.tf_explainer, model)
                else:
                    pass

            
        except FileNotFoundError as e:
            print(
                f"\tProcessor exited with error: '{str(e)}'. Continuing with next session."
            )
            caught_ex = True
            continue
        finally:
            print("...done")

        # Data Saving
        text = Text(data=np.array([json.dumps(data_output)]))

        session_manager : SessionManager
        #session_manager = ss_dataset.sessions[session]['manager']
        #session_manager.output_data_templates["output"] = text
        sm.output_data_templates["output"] = text

        provider.save()

    print("Processing completed!")
    if caught_ex:
        print(
            "Processing job encountered errors for some sessions. Check logs for details."
        )
        exit(1)

# Entry point for nu-explain
def cl_main():
    main(sys.argv[1:])

# Entry point for python
if __name__ == "__main__":
    main(sys.argv[1:])
