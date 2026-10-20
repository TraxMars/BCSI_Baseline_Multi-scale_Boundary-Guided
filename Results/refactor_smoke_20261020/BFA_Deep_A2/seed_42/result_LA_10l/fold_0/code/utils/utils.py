import torch
import numpy as np
from medpy import metric
from scipy import ndimage
import torch.nn.functional as F
import matplotlib.pyplot as plt



def calculate_metric_percase(y_true, y_pred, thr=0.5):
    y_true = y_true.to(torch.float32).cpu().detach().numpy().astype(bool)
    y_pred = (y_pred > thr).astype(bool)

    if y_pred.sum() == 0:
        dice = 0
        jc = 0
        hd = 100
        asd = 100
    else:
        dice = metric.binary.dc(y_pred, y_true)
        jc = metric.binary.jc(y_pred, y_true)
        hd = metric.binary.hd95(y_pred, y_true)
        asd = metric.binary.asd(y_pred, y_true)
    return dice, jc, hd, asd


def boundary_f1_percase(y_true, y_pred, tolerance=1.0):
    """Surface precision/recall F1 at one voxel (no spacing stored in these h5s).

    Surfaces use 6-connected erosion; distance is Euclidean in voxel units.
    Both empty = 1; exactly one empty = 0. This never changes baseline metrics.
    """
    truth = np.asarray(y_true, dtype=bool)
    prediction = np.asarray(y_pred, dtype=bool)
    structure = ndimage.generate_binary_structure(3, 1)
    a = truth & ~ndimage.binary_erosion(truth, structure=structure, border_value=0)
    b = prediction & ~ndimage.binary_erosion(prediction, structure=structure, border_value=0)
    if not a.any() or not b.any():
        return float(not a.any() and not b.any())
    precision = (ndimage.distance_transform_edt(~a)[b] <= tolerance).mean()
    recall = (ndimage.distance_transform_edt(~b)[a] <= tolerance).mean()
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def patients_to_slices(dataset, patiens_num):
    ref_dict = {}
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "PROMISE12" in dataset:
        ref_dict = {"3": 101, "7": 202, }
    elif dataset == "LA" or "LA_Seg_Training" in dataset:
        ref_dict = {"5": 4,"10": 8,"20": 16}
    elif "Pancreas" in dataset:
        ref_dict = {"5": 3, "10": 6, "20": 12, "30": 18}
    elif "BraTS" in dataset:
        ref_dict = {"5": 12,"10": 25, "20": 50}
    elif "Lung" in dataset:
        ref_dict = {"10": 5, "20": 10, "30": 15, "40": 20}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]





