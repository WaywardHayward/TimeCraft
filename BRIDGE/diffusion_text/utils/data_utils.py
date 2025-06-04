import pandas as pd
import numpy as np
from einops import rearrange
from distutils.util import strtobool
from datetime import datetime
from pathlib import Path
from ldm.data.tsg_dataset import load_data_text_from_file
import os
from pathlib import Path

prefix = ''
if 'DATA_ROOT' in os.environ and os.path.exists(os.environ['DATA_ROOT']):
    prefix = Path(os.environ['DATA_ROOT'])
else:
    print("DATA_ROOT not exist or not defined!")

test_data_map = {
    # 'solar': 'solar_{seq_len}_val.npy',
    # 'electricity': 'electricity_{seq_len}_val.npy',
    # 'traffic': 'traffic_{seq_len}_val.npy',
    # 'kddcup': 'kddcup_{seq_len}_val.npy',
    # 'taxi': 'taxi_{seq_len}_val.npy',
    # 'exchange': 'exchange_{seq_len}_val.npy',
    # 'fred_md': 'fred_md_{seq_len}_val.npy',
    # 'nn5': 'nn5_{seq_len}_val.npy',
    # 'web': 'web_{seq_len}_test_sample.npy',
    # 'stock': 'stock_{seq_len}_test_sample.npy',
    # 'temp': 'temp_{seq_len}_val.npy',
    # 'rain': 'rain_{seq_len}_val.npy',
    # 'pedestrian': 'pedestrian_{seq_len}_val.npy',
    # 'wind_4_seconds': 'wind_4_seconds_{seq_len}_val.npy'
}

test_data_map_text = {
    #'solar': 'solar_with_descriptions.csv'
    #'electricity': 'electricity_with_descriptions.csv',
    #'traffic': 'traffic_with_descriptions.csv',
    #'kddcup': 'kddcup_with_descriptions.csv',
    #'taxi': 'taxi_with_descriptions.csv',
    #'exchange': 'exchange_with_descriptions.csv',
    #'fred_md': 'fred_md_with_descriptions.csv',
    'nn5': '{DATA_ROOT}/nn5_numeric_text.npy'
    #'pedestrian': 'pedestrian_with_descriptions.csv',
    #'wind_4_seconds': "wind_with_descriptions.csv",
    #'rain': 'rain_with_descriptions.csv',
    #'temp': 'temp_with_descriptions.csv'
}

import numpy as np
from pathlib import Path

def test_data_loading(data_name, seq_len, stride=1, univar=False, use_text=True):
    global prefix  

    if use_text and data_name in test_data_map_text:
        data_path = test_data_map_text[data_name]

        if '{DATA_ROOT}' in data_path:
            if prefix:
                data_path = data_path.replace('{DATA_ROOT}', str(prefix))
            else:
                raise ValueError("DATA_ROOT is not defined or does not exist!")
        
        if data_path.endswith('.npy'):
            npy_data = np.load(data_path, allow_pickle=True)

            if isinstance(npy_data, np.ndarray) and npy_data.shape == ():
                print("⚠️ `.npy` resolves to empty array, try `.item()` to force conversion...")
                npy_data = npy_data.item()  
            
            if isinstance(npy_data, dict):

                if 'original_numeric' in npy_data and 'original_text' in npy_data:
                    numeric_data = npy_data['original_numeric']
                    text_data = npy_data['original_text']

                    if numeric_data.ndim == 2:
                        numeric_data = numeric_data.reshape(numeric_data.shape[0], numeric_data.shape[1], 1)

                    return numeric_data, text_data
                else:
                    raise ValueError(f"❌ Missing required keys in .npy file: {list(npy_data.keys())}")

            else:
                raise ValueError(f"❌ Unsupported `.npy` format: {type(npy_data)}")


        else:
            num_data, text_embeddings = load_data_text_from_file(data_path, this_window=seq_len)
            print(f"✅ Loaded CSV: num_data.shape={num_data.shape}, text_embeddings.shape={text_embeddings.shape}")
            return num_data, text_embeddings

    elif data_name in test_data_map:
        data_path = prefix / test_data_map[data_name].format(seq_len=seq_len, stride=stride)
        ori_data = np.load(data_path)  # (n, t, c)
        
        if univar:
            ori_data = rearrange(ori_data, 'n t c -> (n c) t 1')
        
        return ori_data

    else:
        raise ValueError(f"Unknown dataset name: {data_name}")



def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )