import tensorflow as tf

def style_conversion(sample, model):
    data = {"success": "failed"}

    counterfactual_dictionary = {}
    counterfactual = model.predict(sample)

    for id, feat in enumerate(counterfactual):
        counterfactual_dictionary[id] = feat

    original_data_dictionary = {}

    for id, feat in enumerate(sample):
        original_data_dictionary[id] = float(feat)

    data = {"success": "true",
        "counterfactual": counterfactual_dictionary,
        "original_data": original_data_dictionary
        }
