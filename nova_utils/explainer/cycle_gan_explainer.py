import tensorflow as tf
import numpy as np

def style_conversion(sample, model):
    data = {"success": "failed"}

    counterfactual_dictionary = {}
    counterfactual = model.predict(np.expand_dims(sample, axis=0))

    for id, feat in enumerate(counterfactual[0]):
        counterfactual_dictionary[id] = float(feat)

    original_data_dictionary = {}

    for id, feat in enumerate(sample):
        original_data_dictionary[id] = float(feat)

    data = {"success": "true",
        "counterfactual": counterfactual_dictionary,
        "original_data": original_data_dictionary
        }
    
    return data