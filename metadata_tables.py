import sys
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import subprocess
cwdir = os.getcwd()
sys.path.append(f'{cwdir}/../erddap_demo')
import utils
import logging
_log = logging.getLogger(__name__)


def write_csv(df, name):
    if not "datasetID" in list(df):
        df["datasetID"] = df.index
    df = df.convert_dtypes()
    _log.info(f"write {name}.csv")
    df.to_csv(f'{cwdir}/output/{name}.csv', sep=';', index=False)
    subprocess.check_call(['/usr/bin/rsync', f'{cwdir}/output/{name}.csv', 'usrerddap@136.243.54.252:/media/data/meta'])
    _log.info(f"sent '{cwdir}/output/{name}.csv to erddap")


if __name__ == '__main__':
    logf = 'metadata_processing.log'
    logging.basicConfig(filename=logf,
                        filemode='a',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    _log.info("Start processing")

    e = utils.init_erddap()

    # Fetch dataset list
    e.response = "csv"
    e.dataset_id = "allDatasets"
    df_datasets = e.to_pandas(parse_dates=['minTime (UTC)', 'maxTime (UTC)'])

    # drop the allDatasets row and make the datasetID the index for easier reading
    df_datasets.set_index("datasetID", inplace=True)
    df_datasets.drop("allDatasets", inplace=True)

    df_datasets = df_datasets[df_datasets.index.str[:3] == "nrt"]
    df_datasets = df_datasets.drop('nrt_SEA057_M75')
    df_datasets = df_datasets.drop('nrt_SEA070_M29')

    #df_datasets = df_datasets.head(3)
    _log.info(f"found {len(df_datasets)} datasets")

    ds_meta = {}
    for dataset_id in tqdm(df_datasets.index):
        ds_meta[dataset_id] = utils.get_meta(dataset_id)

    # Download data
    ds_nrt = utils.download_glider_dataset(df_datasets.index, nrt_only=True)

    # Merge all metadata available in one big column

    df_met = []
    _log.info(f"processing metadata files")
    for dataset_id in df_datasets.index:
        dictrow = {}
        for key, val in ds_meta[dataset_id].items():
            # If the value is a method (like dataset.close) do not include it
            if callable(val):
                continue
            if type(val) is dict:
                if val == {}:
                    continue
                for k, v in val.items():
                    if type(v) is dict:
                        for c, u in v.items():
                            dictrow[f'{k}_{c}'] = u
                    else:
                        dictrow[f'{key}_{k}'] = v
            elif type(val) is str:
                val_rep = val.replace("\n", "")
                dictrow[key] = val_rep
            elif type(val) is list:
                dictrow[key] = str(val)
            else:
                dictrow[key] = val

        dfrow = pd.DataFrame(dictrow, index=[dataset_id])
        df_met.append(dfrow)

    df_met_all = pd.concat(df_met)
    write_csv(df_met_all, 'metadata_table')

    # Merge all variables attributes into one table

    dat_var = []
    _log.info(f"processing attributes files")
    for dataset_id in df_datasets.index[:]:

        d_row = {}
        vars_data = list(ds_nrt[dataset_id].data_vars)
        for i in vars_data[:]:
            att = ds_nrt[dataset_id][i].attrs
            for key, val in att.items():
                if type(val) is list or np.array:
                    d_row[f'{i}_{key}'] = str(val)
                elif str("\n") in str(val):
                    d_row[key] = str(val).replace("\n", "")
                else:
                    d_row[f'{i}_{key}'] = val

        df_row = pd.DataFrame(d_row, index=[dataset_id])
        dat_var.append(df_row)

    var_all = pd.concat(dat_var)
    write_csv(var_all, 'var_attrs_table')
    # Merge the metadata table with the attributes table

    full_table = var_all.merge(df_met_all, left_on=var_all.index, right_on=df_met_all.index)
    write_csv(full_table, 'full_meta_attrs_table')
    _log.info(f"merged metadata and attributes ")
    # Create a smaller, more user friendly table

    table = pd.DataFrame(columns=['glider_serial', 'deployment_id', 'basin', 'deployment_start', 'deployment_end',
                                  'available_variables', 'science_variables', 'ctd', 'oxygen', 'optics', 'ad2cp',
                                  'irradiance', 'nitrate', 'datasetID'])
    missions = df_datasets.index
    dic = ds_meta
    table.deployment_id = range(0, len(missions))
    _log.info(f"Creating users table ")
    for i in range(len(missions)):
        
        d = dic[missions[i]]
        table.glider_serial[i] = f'SEA0{d["glider_serial"]}'
        table.deployment_id[i] = d["deployment_id"]
        table.deployment_start[i] = d["deployment_start"][:10]
        table.deployment_end[i] = d["deployment_end"][:10]
        table.basin[i] = d["basin"]
        table.datasetID[i] = d["dataset_id"]
        table.available_variables[i] = d["variables"]
        table.science_variables[i] = d["variables"]
        table.ctd[i] = d['ctd']
        table.oxygen[i] = d['oxygen']
        table.optics[i] = d['optics']
        if 'irradiance' in d:
            table.irradiance[i] = d['irradiance']
        if 'AD2CP' in d:
            table.ad2cp[i] = d['AD2CP']
        if 'nitrate' in d:
            table.nitrate[i] = d['nitrate']

        nav_var = {'profile_index', 'rowSize', 'latitude', 'longitude', 'time', 'depth',
                   'angular_cmd', 'angular_pos', 'ballast_cmd', 'ballast_pos', 'desired_heading',
                   'dive_num', 'heading', 'internal_pressure', 'internal_temperature', 'linear_cmd',
                   'linear_pos', 'nav_state', 'pitch', 'profile_direction', 'profile_num',
                   'roll', 'security_level', 'vertical_distance_to_seafloor', 'voltage', 'declination'}

        # Iterate each element in list
        # and add them in variable total
        table.science_variables[i] = [i for i in table.science_variables[i] if i not in nav_var]

    write_csv(table, 'users_table')

    _log.info("End processing")
