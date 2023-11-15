import os
import tensorflow as tf
import io
import ast
import numpy as np
import base64
import json
import pickle
import site as s
from PIL import Image
from lime.lime_image import LimeImageExplainer
from lime.lime_tabular import LimeTabularExplainer
from skimage.segmentation import mark_boundaries

def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")
    # resize the input image and preprocess it
    image = image.resize(target)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # return the processed image
    return image

def getTopXpredictions(prediction, topLabels):

    prediction_class = []

    for i in range(0, len(prediction[0])):
        prediction_class.append((i, prediction[0][i]))

    prediction_class.sort(key=lambda x: x[1], reverse=True)

    return prediction_class[:topLabels]

def lime_image(stream_data, frame_id, num_features, top_labels, num_samples, hide_color, hide_rest, positive_only, model):
    data = {"success": "failed"}

    sample = stream_data[frame_id]

    img = prepare_image(Image.fromarray(sample), (224, 224))
    img = img * (1.0 / 255)
    prediction = model.predict(img)
    explainer = LimeImageExplainer()
    img = np.squeeze(img).astype("double")
    explanation = explainer.explain_instance(
        img,
        model.predict,
        top_labels=top_labels,
        hide_color=hide_color == "True",
        num_samples=num_samples,
    )

    top_classes = getTopXpredictions(prediction, top_labels)

    explanations = []

    for cl in top_classes:
        temp, mask = explanation.get_image_and_mask(
            cl[0],
            positive_only=positive_only == "True",
            num_features=num_features,
            hide_rest=hide_rest == "True",
        )
        img_explained = mark_boundaries(temp, mask)
        img = Image.fromarray(np.uint8(img_explained * 255))
        img_byteArr = io.BytesIO()
        img.save(img_byteArr, format="JPEG")
        img_byteArr = img_byteArr.getvalue()
        img64 = base64.b64encode(img_byteArr)
        img64_string = img64.decode("utf-8")

        explanations.append((str(cl[0]), str(cl[1]), img64_string))

    data["explanations"] = explanations
    data["success"] = "success"

    return data


def lime_tabular(stream_data, frame_id, top_class, num_features, model):
    data = {"success": "failed"}

    sample = stream_data[frame_id]
    top_class = getTopXpredictions(model.predict_proba([sample]), 1)
    explainer = LimeTabularExplainer(
        np.asarray(stream_data), mode="classification", discretize_continuous=True
    )
    exp = explainer.explain_instance(
        np.asarray(sample),
        model.predict_proba,
        num_features=num_features,
        top_labels=1,
    )

    explanation_dictionary = {}

    for entry in exp.as_list(top_class[0][0]):
        explanation_dictionary.update({entry[0]: entry[1]})

    data = {"success": "true",
            "explanation": explanation_dictionary}
    return data