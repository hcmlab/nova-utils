"""Utility module to encode and decode specific classes in json

Author:
    Dominik Schiller <dominik.schiller@uni-a.de>
Date:
    14.9.2023

"""

import json

from nova_utils.utils.ssi_xml_utils import Chain, ChainLink, Trainer, ModelIO, URI


class ModelIOEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for ModelIO objects.

    This encoder is used to serialize ModelIO objects to JSON format.

    Attributes:
        None

    """

    def default(self, obj):
        """
        Encodes a ModelIO object to JSON.

        Args:
            obj (ModelIO): The ModelIO object to encode.

        Returns:
            dict: A dictionary representation of the ModelIO object.

        """
        if isinstance(obj, ModelIO):
            return {"type": obj.io_type, "id": obj.io_id, "data": obj.io_data, "default_value": obj.io_default_value}
        return super().default(obj)

class ModelIODecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)
    def object_hook(self, json_obj):
        if json_obj.get("type") and json_obj.get("id") and json_obj.get("data"):
            return ModelIO(json_obj["type"], json_obj["id"], json_obj["data"], json_obj["default_value"])
        else:
            raise ValueError("Invalid JSON format for ModelIO decoding.")


class URIEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for ModelIO objects.

    This encoder is used to serialize ModelIO objects to JSON format.

    Attributes:
        None

    """

    def default(self, obj):
        """
        Encodes a ModelIO object to JSON.

        Args:
            obj (ModelIO): The ModelIO object to encode.

        Returns:
            dict: A dictionary representation of the ModelIO object.

        """
        if isinstance(obj, URI):
            return {"id": obj.uri_id, "url": obj.uri_url, "hash": obj.uri_hash, "tar": obj.uri_tar}
        return super().default(obj)

class URIDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)
    def object_hook(self, json_obj):
        if json_obj.get("id") and json_obj.get("url") and json_obj.get("hash"):
            return ModelIO(json_obj["id"], json_obj["url"], json_obj["hash"], json_obj["tar"])
        else:
            raise ValueError("Invalid JSON format for ModelIO decoding.")



class ChainLinkEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for ChainLink objects.

    This encoder is used to serialize ChainLink objects to JSON format.

    Attributes:
        None

    """

    def default(self, obj):
        """
        Encodes a ChainLink object to JSON.

        Args:
            obj (ChainLink): The ChainLink object to encode.

        Returns:
            dict: A dictionary representation of the ChainLink object.

        """
        if isinstance(obj, ChainLink):
            return {
                "create": obj.create,
                "script": obj.script,
                "optsstr": obj.optsstr,
                "syspath": obj.syspath,
                "tag": obj.tag,
                "multi_role_input": obj.multi_role_input,
            }
        return super().default(obj)


class ChainEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for Chain objects.

    This encoder is used to serialize Chain objects to JSON format.

    Attributes:
        None

    """

    def default(self, obj):
        """
        Encodes a Chain object to JSON.

        Args:
            obj (Chain): The Chain object to encode.

        Returns:
            dict: A dictionary representation of the Chain object.

        """
        if isinstance(obj, Chain):
            return {
                "meta_frame_step": obj.meta_frame_step,
                "meta_left_ctx": obj.meta_left_ctx,
                "meta_right_ctx": obj.meta_right_ctx,
                "meta_backend": obj.meta_backend,
                "meta_description": obj.meta_description,
                "meta_category": obj.meta_category,
                "meta_io": json.dumps(obj.meta_io, cls=ModelIOEncoder),
                "register": obj.register,
                "links": json.dumps(obj.links, cls=ChainLinkEncoder),
            }
        return super().default(obj)


class TrainerEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for Trainer objects.

    This encoder is used to serialize Trainer objects to JSON format.

    Attributes:
        None
    """

    def default(self, obj):
        """
        Encodes a Trainer object to JSON.

        Args:
            obj (Trainer): The Trainer object to encode.

        Returns:
            dict: A dictionary representation of the Trainer object.

        """
        if isinstance(obj, Trainer):
            return {
                "model_script_path": obj.model_script_path,
                "model_option_path": obj.model_option_path,
                "model_option_string": obj.model_optstr,
                "model_weights_path": obj.model_weights_path,
                "model_stream": obj.model_stream,
                "model_create": obj.model_create,
                "model_multirole_input": obj.model_multi_role_input,
                "users": obj.users,
                "classes": obj.classes,
                "streams": obj.streams,
                "register": obj.register,
                "info_trained": obj.info_trained,
                "meta_frame_step": obj.meta_frame_step,
                "meta_right_ctx": obj.meta_right_ctx,
                "meta_left_ctx": obj.meta_left_ctx,
                "meta_balance": obj.meta_balance,
                "meta_backend": obj.meta_backend,
                "meta_io": json.dumps(obj.meta_io, cls=ModelIOEncoder),
                "meta_uri": json.dumps(obj.meta_uri, cls=URIEncoder),
                "meta_description": obj.meta_description,
                "meta_is_iterable": obj.meta_is_iterable,
                "meta_category": obj.meta_category,
                "ssi_v": obj.ssi_v,
                "xml_version": obj.xml_version,
            }
        return super().default(obj)

if __name__ == '__main__':
    from pathlib import Path

    trainer_in_fp = Path(
        r"/Users/dominikschiller/Work/github/nova-server-modules/test/io_test.trainer"
    )
    trainer = Trainer()
    trainer.load_from_file(trainer_in_fp)
    trainer_json = json.dumps(trainer, cls=TrainerEncoder)
    breakpoint()