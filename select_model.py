"""
function: To do model selection in the model folder
return:  model lists including all models in the model folder
"""
import os
import pandas as pd

abs_path = os.path.abspath(__file__)
uppr_dir = os.path.dirname(abs_path)
print(uppr_dir)
model_path = "models/"


def select_model(path):
    lists = [model_name for model_name in os.listdir(path)]
    sorted_list = sorted(lists, key = len,reverse=True)
    return pd.Series(sorted_list)

if __name__ == "__main__":
    lists = select_model(os.path.join(uppr_dir.replace("\\","/")+'/'+model_path))
    print(lists)