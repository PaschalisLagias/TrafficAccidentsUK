import pandas as pd
from os import path


def read_yr_data(yr, folder):
    """
    :param yr:
    :param folder:

    :return:
    """
    accidents_path = path.join(f"{folder}{yr}", f"Accidents{yr}.csv")
    casualties_path = path.join(f"{folder}{yr}", f"Casualties{yr}.csv")
    vehicles_path = path.join(f"{folder}{yr}", f"Vehicles{yr}.csv")

    acc_df = pd.read_csv(accidents_path, low_memory=False)
    cas_df = pd.read_csv(casualties_path, low_memory=False)
    veh_df = pd.read_csv(vehicles_path, low_memory=False)
    return acc_df, cas_df, veh_df


def merge_data():
    pass
