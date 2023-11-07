"""Utility module to load and save xml files written by SSI

Author:
    Dominik Schiller <dominik.schiller@uni-a.de>
Date:
    14.9.2023

"""
import xml.etree.ElementTree as ET
from pathlib import Path

from nova_utils.utils.string_utils import string_to_bool


class ModelIO:
    """
     ModelIO defines the inputs and outputs of a model in a chain or trainer file.

    This class is used to create and work with ChainLinks.

    Attributes:
        io_type (str): Defines if the object describes an input or an output. Can be either "input" or "output"
        io_id (str): String for the model to identify the input
        io_data(str): Description of the data <data_class>:<data_type>:<specific_data_type>

            ``"data_class"``
                     General data class. Either "annotation" or "stream"
            ``"data_type"``
                     Type of the data_class. Matches classnames data type from nova_utils.data .
                     "Video", "Audio", "SSIStream" for streams. "Discrete", "Free" or "Continuous" for annotations.
            ``"specific_data_type"``
                    Optional string identifier to specify either the annotation scheme or the type of feature. E.g. "transcript" or "Openface"

    Args:
        io_type (str): Defines if the object describes an input or an output. Can be either "input" or "output"
        io_id (str): String for the model to identify the input
        io_data(str): Description of the data <data_class>:<data_type>:<specific_data_type>

            ``"data_class"``
                     General data class. Either "annotation" or "stream"
            ``"data_type"``
                     Type of the data_class. Matches classnames data type from nova_utils.data .
                     "Video", "Audio", "SSIStream" for streams. "Discrete", "Free" or "Continuous" for annotations.
            ``"specific_data_type"``
                    Optional string identifier to specify either the annotation scheme or the type of feature. E.g. "transcript" or "Openface"
        io_default_value (str): Default value for stream name or annotation scheme to read from or write to
    """

    def __init__(
        self,
        io_type: str,
        io_id: str,
        io_data: str,
        io_default_value: str,
    ):

        """
        Initialize a ModelIO object with the specified parameters.

        """
        self.io_type = io_type
        self.io_id = io_id
        self.io_data = io_data
        self.io_default_value = io_default_value

class URI:
    """
 URI objects are describing additional resources that are required by a module.

Attributes:
    uri_id (str): Defines if the object describes an input or an output. Can be either "input" or "output"
    uri_url (str): String for the model to identify the input
    uri_hash(str): Md5 hash to verify data integrity
    uri_tar(bool): If the uri describes a tarball

Args:
    uri_id (str): Defines if the object describes an input or an output. Can be either "input" or "output"
    uri_url (str, optional): String for the model to identify the input
    uri_hash(str, optional): Md5 hash to verify data integrity
    uri_tar(bool, optional): If the uri describes a tarball
"""

    def __init__(
            self,
            uri_id: str,
            uri_url: str = None,
            uri_hash: str = None,
            uri_tar: bool = False,
    ):

        """
        Initialize a ModelIO object with the specified parameters.

        """
        self.uri_id = uri_id
        self.uri_url = uri_url
        self.uri_hash = uri_hash
        self.uri_tar = uri_tar

class Trainer:
    """
    Class for representing and working with Trainer configuration.

    This class is used to create, load, and write Trainer configurations in XML format.

    Attributes:
        model_script_path (str): Path to the model script file.
        model_option_path (str): Path to the model option file.
        model_optstr (str): Model option string.
        model_weights_path (str): Path to the model weights file.
        model_stream (int): Model stream identifier.
        model_create (str): Model creation type.
        model_multi_role_input (bool): Indicates if the model accepts multi-role input.
        users (list): List of user configurations.
        classes (list): List of class configurations.
        streams (list): List of stream configurations.
        register (list): List of utilized dlls.
        info_trained (bool): Indicates if the Trainer has been trained.
        meta_frame_step (int): Frame step  for the Trainer.
        meta_right_ctx (int): Right context size for the Trainer.
        meta_left_ctx (int): Left context size for the Trainer.
        meta_balance (str): Balance type for the Trainer.
        meta_is_iterable (str): Bool that indicates if the module requires data processing via nova-server iterator.
        meta_is_processable (str): Bool that indicates if the implements the Processor interface.
        meta_is_trainable (str): Bool that indicates if the implements the Trainer interface.
        meta_is_explainable (str): Bool that indicates if the module supports the Explain
        meta_backend (str): Backend of the model. E.g. sklearn, pytorch, tensorflow. Used to explain and train the model.
        meta_category (str): Category of the trainer.
        meta_description (str): Description of the trainer.
        meta_io(list[ModelIO]): Description of the inputs and outputs of the model.
        meta_uri(list[URI]): Description of additional resources required by the model.
        ssi_v (str): SSI version.
        xml_version (str): XML version.

    Args:
        model_script_path (str, optional): Path to the model script file. Defaults to empty string.
        model_option_path (str, optional): Path to the model option file. Defaults to empty string.
        model_option_string (str, optional): Model option string. Defaults to empty string.
        model_weights_path (str, optional): Path to the model weights file. Defaults to empty string.
        model_stream (int, optional): Model stream identifier. Default is 0.
        model_create (str, optional): Model creation type. Default is "PythonModel".
        model_multirole_input (bool, optional): Indicates if the model supports multi-role input. Default is False.
        users (list, optional): List of user information. Default is None.
        classes (list, optional): List of class information. Default is None.
        streams (list, optional): List of stream information. Default is None.
        register (list, optional): List of registered items. Default is None.
        info_trained (bool, optional): Indicates if the model is trained. Default is False.
        meta_is_iterable (str, optional): Bool that indicates if the module requires data processing via nova-server iterator. Defaults to False.
        meta_is_processable (str, optional): Bool that indicates if the implements the Processor interface. Defaults to True.
        meta_is_trainable (str, optional): Bool that indicates if the implements the Trainer interface. Defaults to False.
        meta_is_explainable (str, optional): Bool that indicates if the module supports the Explain. Defaults to False.
        meta_frame_step (int, optional): Frame step value for metadata. Default is 0.
        meta_right_ctx (int, optional): Right context value for metadata. Default is 0.
        meta_left_ctx (int, optional): Left context value for metadata. Default is 0.
        meta_balance (str, optional): Balance type for metadata. Default is "none".

        meta_backend (str): Backend of the model. E.g. sklearn, pytorch, tensorflow. Used to explain and train the model. Defaults to 'unknown'.
        meta_category (str, optional): Category of the trainer. Default is "".
        meta_description (str, optional): Description of the trainer. Default is "".
        meta_io(list[ModelIO], optional): Description of the inputs and outputs of the model. Defaults to None.
        meta_uri(list[URI], optional): Description of additional resources required by the model.
        ssi_v (str, optional): SSI version. Default is "5".
        xml_version (str, optional): XML version. Default is "1.0".

    """

    def __init__(
        self,
        model_script_path: str = "",
        model_option_path: str = "",
        model_option_string: str = "",
        model_weights_path: str = "",
        model_stream: int = 0,
        model_create: str = "PythonModel",
        model_multirole_input=False,
        users: list = None,
        classes: list = None,
        streams: list = None,
        register: list = None,
        info_trained: bool = False,
        meta_frame_step: int = 0,
        meta_right_ctx: int = 0,
        meta_left_ctx: int = 0,
        meta_balance: str = "none",
        meta_backend: str = "unknown",
        meta_description: str = "",
        meta_category: str = "",
        meta_is_iterable: bool = False,
        meta_is_processable: bool = True,
        meta_is_trainable: bool = False,
        meta_is_explainable: bool = False,
        meta_io: list[ModelIO] = None,
        meta_uri: list[URI] = None,
        ssi_v="5",
        xml_version="1.0",
    ):
        """
        Initialize a Trainer object with various parameters.

        """

        self.model_multi_role_input = None
        self.model_script_path = model_script_path
        self.model_option_path = model_option_path
        self.model_optstr = model_option_string
        self.model_weights_path = model_weights_path
        self.model_stream = model_stream
        self.model_create = model_create
        self.users = users if users is not None else []
        self.classes = classes if classes is not None else []
        self.streams = streams if streams is not None else []
        self.register = register if register is not None else []
        self.info_trained = info_trained
        self.meta_frame_step = meta_frame_step
        self.meta_right_ctx = meta_right_ctx
        self.meta_left_ctx = meta_left_ctx
        self.meta_balance = meta_balance
        self.meta_backend = meta_backend
        self.meta_description = meta_description
        self.meta_category = meta_category
        self.meta_is_trainable = meta_is_trainable
        self.meta_is_explainable = meta_is_explainable
        self.meta_is_iterable = meta_is_iterable
        self.meta_io = meta_io if meta_io is not None else []
        self.meta_uri = meta_uri if meta_uri is not None else []
        self.ssi_v = ssi_v
        self.xml_version = xml_version
        self.model_multi_role_input = model_multirole_input

    def load_from_file(self, fp):
        """
        Load Trainer configuration from an XML file.

        Args:
            fp (str or Path): The file path to the XML file.

        """
        root = ET.parse(Path(fp))
        info = root.find("info")
        meta = root.find("meta")
        register = root.find("register")
        streams = root.find("streams")
        classes = root.find("classes")
        users = root.find("users")
        model = root.find("model")

        if info is not None:
            self.info_trained = string_to_bool(info.get("trained", ""))
        if meta is not None:
            self.meta_left_ctx = meta.get("leftContext", default="0")
            self.meta_right_ctx = meta.get("rightContext", default="0")
            self.meta_frame_step = meta.get("frameStep", default="0")
            self.meta_balance = meta.get("balance", default="none")
            self.meta_backend = meta.get("backend", default="Python")
            self.meta_description = meta.get("description", default="")
            self.meta_category = meta.get("category", default="")
            self.meta_is_iterable = meta.get("is_iterable", default="True")
            self.meta_is_processable = meta.get("is_processable", default="True")
            self.meta_is_trainable = meta.get("is_trainable", default="False")
            self.meta_is_explainable = meta.get("is_explainable", default="False")

            for io_tag in meta.findall("io"):
                self.meta_io.append(
                    ModelIO(io_tag.get("type"), io_tag.get("id"), io_tag.get("data"), io_tag.get("default_value"))
                )

            for uri_tag in meta.findall("uri"):
                self.meta_uri.append(
                    URI( uri_tag.get("id"), uri_tag.get("url"), uri_tag.get("hash"), string_to_bool(uri_tag.get("tar", "")))
                )
        if register is not None:
            for r in register:
                self.register.append(r.attrib)
        if streams is not None:
            for s in streams:
                self.streams.append(s.attrib)
        if classes is not None:
            for c in classes:
                self.classes.append(c.attrib)
        if users is not None:
            for u in users:
                self.users.append(u.attrib)
        if model is not None:
            self.model_stream = model.get("stream", "0")
            self.model_create = model.get("create", "PythonModel")
            self.model_option_path = model.get("option", "")
            self.model_script_path = model.get("script", "")
            self.model_weights_path = model.get("path", "")
            self.model_optstr = model.get("optstr", "")
            self.model_multi_role_input = string_to_bool( model.get("multi_role_input", "") )


    def write_to_file(self, fp):
        """
        Write Trainer configuration to an XML file.

        Args:
            fp (str or Path): The file path to save the XML file.

        """
        root = ET.Element("trainer")
        ET.SubElement(root, "info", trained=str(self.info_trained))
        meta = ET.SubElement(
            root,
            "meta",
            frameStep=str(self.meta_frame_step),
            leftContext=str(self.meta_left_ctx),
            rightContex=str(self.meta_right_ctx),
            balance=self.meta_balance,
            backend=self.meta_backend,
            category=self.meta_category,
            description=self.meta_description,
            meta_is_iterable = str(self.meta_is_iterable),
            meta_is_trainable = str(self.meta_is_trainable),
            meta_is_explainable = str(self.meta_is_explainable)
        )

        io: ModelIO
        for io in self.meta_io:
            ET.SubElement(
                meta,
                "io",
                id=io.io_id,
                type=io.io_type,
                data=io.io_data,
                default_value=io.io_default_value
            )

        uri: URI
        for uri in self.meta_uri:
            ET.SubElement(
                meta,
                "uri",
                id=uri.uri_id,
                url=uri.uri_url,
                hash=uri.uri_hash,
                tar=str(uri.uri_tar)
            )


        register = ET.SubElement(root, "register")
        for r in self.register:
            ET.SubElement(register, "item", **r)
        streams = ET.SubElement(root, "streams")
        for s in self.streams:
            ET.SubElement(streams, "item", **s)
        classes = ET.SubElement(root, "classes")
        for c in self.classes:
            ET.SubElement(classes, "item", **c)
        users = ET.SubElement(root, "users")
        for u in self.users:
            ET.SubElement(users, "item", **u)
        ET.SubElement(
            root,
            "model",
            create=self.model_create,
            stream=str(self.model_stream),
            path=self.model_weights_path,
            script=self.model_script_path,
            optstr=self.model_optstr,
            option=self.model_option_path,
        )

        tree = ET.ElementTree(root)
        ET.indent(tree, space="    ", level=0)

        if not fp.suffix:
            fp = fp.with_suffix(".trainer")
        tree.write(fp)


class ChainLink:
    """
    Class for representing single steps of a SSI chain configuration.

    This class is used to create and work with ChainLinks.

    Attributes:
        create (str): ChainLink type.
        script (str): ChainLink script.
        optsstr (str): ChainLink options string.
        syspath (str): ChainLink system path.
        tag (str): ChainLink tag.
        multi_role_input (bool): Indicates if the ChainLink accepts multi-role input.

    Args:
        create (str): ChainLink type.
        script (str): ChainLink script.
        optsstr (str): ChainLink options string.
        syspath (str): ChainLink system path.
        tag (str, optional): ChainLink tag. Defaults tu "feature"
        multi_role_input (str, optional): Indicates if the ChainLink accepts multi-role input. Defaults to False.

    """

    def __init__(
        self,
        create: str,
        script: str,
        optsstr: str,
        syspath: str,
        tag: str = "feature",
        multi_role_input: str = "False",
        **kwargs,
    ):
        """
        Initialize a ChainLink object with the specified parameters.

        """
        self.create = create
        self.script = script
        self.optsstr = optsstr
        self.syspath = syspath
        self.tag = tag
        self.multi_role_input = True if multi_role_input == "True" else False


class Chain:
    """
    Class for representing and working with Chain configuration.

    This class is used to create, load, and write Chain configurations in XML format.

    Attributes:
        meta_frame_step (str): Meta frame step value.
        meta_left_ctx (str): Meta left context value.
        meta_right_ctx (str): Meta right context value.
        meta_backend (str): Backend type for the Chain.
        meta_description (str): Description for the Chain.
        meta_category (str): Category for the Chain.
        meta_io(list[ModelIO], optional): Description of the inputs and outputs of the model.
        register (list): List of register configurations.
        links (list): List of ChainLink configurations.

    Args:
        meta_frame_step (str, optional): Meta frame step information. Defaults to empty string.
        meta_left_context (str, optional): Left context metadata. Defaults to empty string.
        meta_right_context (str, optional): Right context metadata. Defaults to empty string.
        meta_backend (str, optional): Backend type for metadata. Default is "nova-server".
        meta_description (str, optional): Description for metadata. Defaults to empty string.
        meta_category (str, optional): Category for metadata. Defaults to empty string.
        meta_io(list[ModelIO], optional): Description of the inputs and outputs of the model. Defaults to None.
        register (list, optional): List of registered items. Default is None.
        links (list, optional): List of ChainLink objects. Default is None.

    """

    def __init__(
        self,
        meta_frame_step: str = "",
        meta_left_context: str = "",
        meta_right_context: str = "",
        meta_backend: str = "nova-server",
        meta_description: str = "",
        meta_category: str = "",
        meta_io: list[ModelIO] = None,
        register: list = None,
        links: list = None,
    ):
        """
        Initialize a Chain object with the specified parameters.

        """
        self.meta_frame_step = meta_frame_step
        self.meta_left_ctx = meta_left_context
        self.meta_right_ctx = meta_right_context
        self.meta_backend = meta_backend
        self.meta_description = meta_description
        self.meta_category = meta_category
        self.meta_io = meta_io if meta_io is not None else []
        self.register = register if register else []
        self.links = links if links else []

    def load_from_file(self, fp):
        """
        Load Chain configuration from an XML file.

        Args:
            fp (str or Path): The file path to the XML file.

        """
        tree = ET.parse(Path(fp))
        root = tree.getroot()
        meta = tree.find("meta")
        register = tree.find("register")
        links = []
        for child in root:
            if child.tag == "feature" or child.tag == "filter":
                links.append(child)

        if meta is not None:
            self.meta_frame_step = meta.attrib.get("frameStep", "0")
            self.meta_left_ctx = meta.attrib.get("leftContext", "0")
            self.meta_right_ctx = meta.attrib.get("rightContext", "0")
            self.meta_backend = meta.attrib.get("backend", "nova-server")
            self.meta_description = meta.attrib.get("description", "")
            self.meta_category = meta.attrib.get("category", "")
            for io_tag in meta.findall("io"):
                self.meta_io.append(
                    ModelIO(io_tag.get("type"), io_tag.get("id"), io_tag.get("data"), io_tag.get("default_value"))
                )

        if register is not None:
            for r in register:
                self.register.append(r.attrib)

        for link in links:
            item = link.find("item")
            new_link = ChainLink(**item.attrib, tag=link.tag)
            self.links.append(new_link)

    def write_to_file(self, fp):
        """
        Write Chain configuration to an XML file.

        Args:
            fp (str or Path): The file path to save the XML file.

        """
        root = ET.Element("chain")
        meta = ET.SubElement(
            root,
            "meta",
            frameStep=str(self.meta_frame_step),
            leftContext=str(self.meta_left_ctx),
            rightContex=str(self.meta_right_ctx),
            backend=str(self.meta_backend),
            description=str(self.meta_description),
            category=str(self.meta_category),
        )

        io: ModelIO
        for io in self.meta_io:
            ET.SubElement(
                meta,
                "io",
                id=io.io_id,
                type=io.io_type,
                data=io.io_data
            )

        register = ET.SubElement(root, "register")
        for r in self.register:
            ET.SubElement(register, "item", **r)



        cl: ChainLink
        for cl in self.links:
            link = ET.SubElement(root, cl.tag)
            ET.SubElement(
                link,
                "item",
                create=cl.create,
                script=cl.script,
                syspath=cl.syspath,
                optsstr=cl.optsstr,
                multi_role_input=str(cl.multi_role_input),
            )

        tree = ET.ElementTree(root)
        ET.indent(tree, space="    ", level=0)

        if not fp.suffix:
            fp = fp.with_suffix(".chain")
        tree.write(fp)


if __name__ == "__main__":

    trainer_in_fp = Path(
        r"C:\Users\schildom\Documents\github\nova-server-modules\demo\uc1\uc1.trainer"
    )
    trainer_out_fp = Path("test_trainer.trainer")

    trainer = Trainer()
    trainer.load_from_file(trainer_in_fp)

    trainer.write_to_file(trainer_out_fp)

    chain_in_fp = Path(
        "/Users/dominikschiller/Work/github/nova-server-modules/chains/nova-server/utility/uc1/uc1.chain"
    )
    chain_out_fp = Path("test_chain.chain")

    chain = Chain()
    chain.load_from_file(chain_in_fp)
    chain.write_to_file(chain_out_fp)
    breakpoint()
