"""Utility module to encode and decode specific classes in json

Author:
    Dominik Schiller <dominik.schiller@uni-a.de>
Date:
    14.9.2023

"""

import json
from nova_utils.utils.ssi_xml_utils import Chain,ChainLink, Trainer

class ChainLinkEncoder(json.JSONEncoder):
    def default(self, obj):
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
    def default(self, obj):
        if isinstance(obj, Chain):
            return {
                "meta_frame_step": obj.meta_frame_step,
                "meta_left_ctx": obj.meta_left_ctx,
                "meta_right_ctx": obj.meta_right_ctx,
                "meta_backend": obj.meta_backend,
                "meta_description": obj.meta_description,
                "meta_category": obj.meta_category,
                "register": obj.register,
                "links": json.dumps(obj.links, cls=ChainLinkEncoder),
            }
        return super().default(obj)

class TrainerEncoder(json.JSONEncoder):
    def default(self, obj):
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
                "meta_right_ctx": obj.meta_right_ctx,
                "meta_left_ctx": obj.meta_left_ctx,
                "meta_balance": obj.meta_balance,
                "meta_backend": obj.meta_backend,
                "ssi_v": obj.ssi_v,
                "xml_version": obj.xml_version,
            }
        return super().default(obj)
