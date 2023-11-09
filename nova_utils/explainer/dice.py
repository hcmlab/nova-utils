import pandas as pd
import dice_ml
import numpy as np


def dice_explain(stream_data, anno, frame, dim, anno_scheme, ml_backend, class_counterfactual, num_counterfactuals, model):
    data = {"success": "failed"}
    train_data_available = False

    # ADD logic to display multiple counterfactuals
    num_counterfactuals = 10
    # num_counterfactuals = int(request.get("numCounterfactuals"))

    sample = stream_data[frame]
    scheme_type = anno_scheme.scheme_type.name
    anno_name = anno_scheme.name

    if scheme_type == "DISCRETE":
    
        classes_dict = anno_scheme.classes

        for i, a in enumerate(anno):
            anno[i] = np.argmax(a)

        train_data_available = True
    else:
        pass

    if ml_backend == "SKLEARN":
        model = dice_ml.Model(model=model, backend="sklearn")
    elif ml_backend == "TENSORFLOW":
        pass
    elif ml_backend == "PYTORCH":
        pass

    dim_cols = list(range(dim))
    stream_data_as_data_frame = pd.DataFrame(stream_data, columns=dim_cols)
    stream_data_as_data_frame_and_anno = pd.DataFrame(stream_data, columns=dim_cols)
    stream_data_as_data_frame_and_anno[anno_name] = anno

    if not train_data_available:
        feature_dic = {}

        for feat in dim_cols:
            # add min max range for models without train data
            feature_dic[str(feat)] = [0, 1]

        d = dice_ml.Data(features=feature_dic, outcome_name=anno_name)
    else:
        d = dice_ml.Data(dataframe=stream_data_as_data_frame_and_anno, continuous_features=dim_cols, outcome_name=anno_name)

    sample_dic = {}

    for id, s in enumerate(sample):
        sample_dic[str(id)] = s

    exp = dice_ml.Dice(d, model, method="genetic")
    try:
        dice_exp = exp.generate_counterfactuals(stream_data_as_data_frame[frame:frame+1], total_CFs=num_counterfactuals, desired_class=int(class_counterfactual), initialization="random")

        explanation_dictionary = {}
        counterfactual = dice_exp.cf_examples_list[0].final_cfs_df_sparse.drop(anno_name, axis=1).to_numpy()[0]

        for id, feat in enumerate(counterfactual):
            explanation_dictionary[id] = feat

        imp = {}

        if num_counterfactuals >= 10:
            # needs at least 10 cf
            imp = exp.local_feature_importance(stream_data_as_data_frame[frame:frame+1], cf_examples_list=dice_exp.cf_examples_list)
        else:
            # no cf examples
            imp = exp.local_feature_importance(stream_data_as_data_frame[frame:frame+1], posthoc_sparsity_param=None, desired_class=int(class_counterfactual))
        global_imp = exp.global_feature_importance(stream_data_as_data_frame[0:10], total_CFs=10, posthoc_sparsity_param=None, desired_class=int(class_counterfactual))

        data = {"success": "true",
                "explanation": explanation_dictionary,
                "local_importance": imp.local_importance[0],
                "global_importance": global_imp.summary_importance}
    except:
        data = {"sucess": "failed"}

    return data
