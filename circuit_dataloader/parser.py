import os
import re
import torch
from copy import deepcopy
from collections import OrderedDict
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Tuple, List
import pandas as pd



def parse_blk_tml(circuit:str, area_util:float, root:str="data_openroad") -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], float, float]:
    """
    blk_wh_dict: {
        "[block_name]": {
            "w": float,
            "h": float,
        }
    }

    tml_xy_dict: {
        "[terminal_name]": {
            "x": float,
            "y": float,
        }
    }

    Also return  the width and height of the chip.
    """

    df_blk = pd.read_csv(os.path.join(root, f"{circuit}.blk.csv"), dtype={'name':str, 'w':float, 'h':float, 'x':float, 'y':float, 'z':int, 'preplaced':int})
    #df_blk = pd.read_csv(os.path.join(root, f"{circuit}_2.blk.csv"), dtype={'name':str, 'w':float, 'h':float, 'x':float, 'y':float, 'z':int, 'preplaced':int})
    blk_wh_dict = {}
    for row in df_blk.itertuples():
        name = row.name
        w = row.w
        h = row.h
        blk_wh_dict[name] = {
            'w': w,
            'h': h,
            'virtual': False,
        }

        # optional fields
        # "x", "y", "z"
        optional_fields = ['x', 'y', 'z', 'preplaced']
        for field in optional_fields:
            if hasattr(row, field):
                blk_wh_dict[name][field] = getattr(row, field)

    blk_wh_dict = OrderedDict(sorted(blk_wh_dict.items()))
    blk_area = (df_blk['w'] * df_blk['h']).sum()
    die_area = blk_area / area_util
    # large chip_w, chip_h area
    chip_w = chip_h = die_area ** 0.5
    print("die area: ", die_area)
    print("blk_area: ", blk_area)
    print("chip_w: ", chip_w, "chip_h: ", chip_h)

    df_tml = pd.read_csv(os.path.join(root, f"{circuit}.tml.csv"), dtype={'name':str, 'x':float, 'y':float})
    
    tml_xy_dict = {}
    for row in df_tml.itertuples():
        name = row.name
        x = row.x
        y = row.y
        tml_xy_dict[name] = {
            'x': x,
            'y': y,
        }
    tml_xy_dict = OrderedDict(sorted(tml_xy_dict.items()))

    # print("blk_wh_dict: ", len(blk_wh_dict))
    # print("tml_xy_dict: ", len(tml_xy_dict))               
    
    return blk_wh_dict, tml_xy_dict, float(chip_w), float(chip_h)

def parse_blk_xyz(circuit: str, root:str="data_openroad")->Dict[str, List[float]]:
    """
    Return the (x,y,z) of the blocks given the fp.txt
    """
    blk_xyz_dict = {}
    path = os.path.join(root, f"{circuit}.fp.txt")
    # path = os.path.join(root, f"{circuit}_2.fp.txt")
    with open(path, "r") as f:
        for line in f.readlines():
            line_list = line.strip("\n").split(",")
            blk_name = line_list[0]
            blk_x, blk_y, blk_z = float(line_list[1]), float(line_list[2]), float(line_list[-1])
            blk_xyz_dict[blk_name] = [blk_x, blk_y, blk_z]
    return blk_xyz_dict


def parse_net(circuit:str, root:str="data_openroad") -> List[List[str]]:
    """
    Return netlist, each net is a list consisting of str, the name of pin.
    """
    path = os.path.join(root, f"{circuit}.net.csv")
    df_net = pd.read_csv(path, dtype={'net':str})
    nets = []
    for row in df_net.itertuples():
        net = eval(row.net)
        nets.append(net)
    return nets



def map_tml(tml_xy_dict:OrderedDict, chip_w:float, chip_h:float) -> Dict[str, Dict[str, float]]:
    if len(tml_xy_dict) == 0:
        print("[INFO] No terminal in the circuit")
        return tml_xy_dict
    
    tml_xy = []
    for tml_name in tml_xy_dict:
        x, y = tml_xy_dict[tml_name]['x'], tml_xy_dict[tml_name]['y']
        tml_xy.append([x, y])
    tml_xy = torch.Tensor(tml_xy)

    print("[INFO] original termimal x range: [{}, {}]".format(tml_xy[:,0].min().item(), tml_xy[:,0].max().item()))
    print("[INFO] original termimal y range: [{}, {}]".format(tml_xy[:,1].min().item(), tml_xy[:,1].max().item()))
    print("[INFO] new termimal x range: [{}, {}]".format(0, chip_w))
    print("[INFO] new termimal y range: [{}, {}]".format(0, chip_h))


    xy = tml_xy.detach().cpu().numpy()
    # chip_h, chip_w must be the same
    scaler_x = MinMaxScaler((0, chip_w))
    scaler_y = MinMaxScaler((0, chip_h))
    x = scaler_x.fit_transform(xy[:,0:1])
    y = scaler_y.fit_transform(xy[:,1:2])

    normalized_tml_xy_dict = OrderedDict()
    for i, tml_name in enumerate(tml_xy_dict):
        normalized_tml_xy_dict[tml_name] = {
            'x': x[i][0],
            'y': y[i][0],
        }
    return normalized_tml_xy_dict
