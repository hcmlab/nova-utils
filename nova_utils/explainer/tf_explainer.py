import os
import tensorflow as tf
import tf_explain
import ast
import numpy as np
import base64
from PIL import Image
from PIL import Image as pilimage
import io as inputoutput

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

def tf_explainer(sample, explainer, model):
    data = {"success": "failed"}

    image = prepare_image(Image.fromarray(sample), (224, 224))
    image = image * (1.0 / 255)
    prediction = model.predict(image)
    topClass = getTopXpredictions(prediction, 1)
    print(topClass[0])
    image = np.squeeze(image)

    # NOTE GradCam is broken "CRITICAL Job failed with exception 'KerasTensor' object has no attribute 'node'" likely keras version problem
    if explainer == "GRADCAM":
        im = ([image], None)
        from tf_explain.core.grad_cam import GradCAM

        exp = GradCAM()
        imgFinal = exp.explain(im, model, class_index=topClass[0][0])
        # exp.save(imgFinal, ".", "grad_cam.png")

    elif explainer == "OCCLUSIONSENSITIVITY":
        im = ([image], None)
        from tf_explain.core.occlusion_sensitivity import OcclusionSensitivity

        exp = OcclusionSensitivity()
        imgFinal = exp.explain(
            im, model, class_index=topClass[0][0], patch_size=10
        )
        # exp.save(imgFinal, ".", "grad_cam.png")

    elif explainer == "GRADIENTSINPUTS":
        im = (np.array([image]), None)
        from tf_explain.core.gradients_inputs import GradientsInputs

        exp = GradientsInputs()
        imgFinal = exp.explain(im, model, class_index=topClass[0][0])
        # exp.save(imgFinal, ".", "gradients_inputs.png")

    elif explainer == "VANILLAGRADIENTS":
        im = (np.array([image]), None)
        from tf_explain.core.vanilla_gradients import VanillaGradients

        exp = VanillaGradients()
        imgFinal = exp.explain(im, model, class_index=topClass[0][0])
        # exp.save(imgFinal, ".", "gradients_inputs.png")

    elif explainer == "SMOOTHGRAD":
        im = (np.array([image]), None)
        from tf_explain.core.smoothgrad import SmoothGrad

        exp = SmoothGrad()
        imgFinal = exp.explain(im, model, class_index=topClass[0][0])
        # exp.save(imgFinal, ".", "gradients_inputs.png")

    elif explainer == "INTEGRATEDGRADIENTS":
        im = (np.array([image]), None)
        from tf_explain.core.integrated_gradients import IntegratedGradients

        exp = IntegratedGradients()
        imgFinal = exp.explain(im, model, class_index=topClass[0][0])
        # exp.save(imgFinal, ".", "gradients_inputs.png")

    elif explainer == "ACTIVATIONVISUALIZATION":
        # need some solution to find out and submit layers name
        im = (np.array([image]), None)
        from tf_explain.core.activations import ExtractActivations

        exp = ExtractActivations()
        imgFinal = exp.explain(im, model, layers_name=["activation_1"])
        # exp.save(imgFinal, ".", "gradients_inputs.png")

    img = pilimage.fromarray(imgFinal)
    imgByteArr = inputoutput.BytesIO()
    img.save(imgByteArr, format="JPEG")
    imgByteArr = imgByteArr.getvalue()

    img64 = base64.b64encode(imgByteArr)
    img64_string = img64.decode("utf-8")

    data["explanation"] = img64_string
    data["prediction"] = str(topClass[0][0])
    data["prediction_score"] = str(topClass[0][1])
    data["success"] = "success"

    return data