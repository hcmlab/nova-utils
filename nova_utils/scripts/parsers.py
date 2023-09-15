"""parsers.py - Common parsers for execution scripts

Author:
    Dominik Schiller <dominik.schiller@uni-a.de>
Date:
    06.09.2023

This module defines argparse parsers for configuring the connection to the NOVA-DB and creating a NovaIterator.

"""

import argparse
import json

# Parser for NOVA database connection
nova_db_parser = argparse.ArgumentParser(
    description="Parse Information required to connect to the NOVA-DB", add_help=False
)
nova_db_parser.add_argument(
    "--db_host", type=str, required=True, help="The ip-address of the NOVA-DB server"
)
nova_db_parser.add_argument(
    "--db_port", type=int, required=True, help="The ip-address of the NOVA-DB server"
)
nova_db_parser.add_argument(
    "--db_user",
    type=str,
    required=True,
    help="The user to authenticate at the NOVA-DB server",
)
nova_db_parser.add_argument(
    "--db_password",
    type=str,
    required=True,
    help="The password for the NOVA-DB server user",
)

# Parser for NOVA iterator
nova_iterator_parser = argparse.ArgumentParser(
    description="Parse Information required to create a NovaIterator", add_help=False
)
nova_iterator_parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    help="Name of the dataset. Must match entries in NOVA-DB",
)
nova_iterator_parser.add_argument(
    "--data_dir",
    type=str,
    required=True,
    help="Path to the NOVA data directory using Windows UNC-Style",
)
nova_iterator_parser.add_argument(
    "--sessions",
    type=json.loads,
    required=True,
    help="Json formatted List of sessions to apply the iterator to",
)
nova_iterator_parser.add_argument(
    "--data",
    type=json.loads,
    required=True,
    help="Json formatted String containing dictionaries that describe the data to load",
)
nova_iterator_parser.add_argument(
    "--frame_size",
    type=str,
    help="Size of the data frame measured in time. Defaults to None",
)
nova_iterator_parser.add_argument(
    "--start",
    type=str,
    help="Start time for processing measured in time. Defaults to None",
)
nova_iterator_parser.add_argument(
    "--end", type=str, help="End time for processing measured in time. Defaults to None"
)
nova_iterator_parser.add_argument(
    "--left_context",
    type=str,
    help="Left context duration measured in time. Defaults to None",
)
nova_iterator_parser.add_argument(
    "--right_context",
    type=str,
    help="Stride for iterating over data measured in time. If stride is not set explicitly it will be set to frame_size. Defaults to None",
)
nova_iterator_parser.add_argument(
    "--stride",
    type=str,
    help="Stride for iterating over data measured in time. If stride is not set explicitly it will be set to frame_size. Defaults to None",
)
nova_iterator_parser.add_argument(
    "--add_rest_class",
    type=str,
    help="Whether to add a rest class for discrete annotations. Defaults to True",
)
nova_iterator_parser.add_argument(
    "--fill_missing_data",
    type=str,
    help="Whether to fill missing data. Defaults to True",
)

# Parser for NOVA-Server module
nova_server_module_parser = argparse.ArgumentParser(
    description="Parse Information required to execute a NOVA-Server module",
    add_help=False,
)
nova_server_module_parser.add_argument(
    "--cml_dir", type=str, help="CML-Base directory for the NOVA-Server module"
)
nova_server_module_parser.add_argument(
    "--opt_str",
    type=str,
    help="Json formatted String containing dictionaries with key value pairs, setting the options for a NOVA-Server module",
)