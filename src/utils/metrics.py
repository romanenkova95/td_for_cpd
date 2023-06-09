"""Functions for metrics calculation."""
from collections import defaultdict
from typing import List, Sequence, Tuple, Optional
import gc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
#import utils.kl_cpd as klcpd
from time import time


def find_first_change(mask: np.array) -> np.array:
    """Find first change in batch of predictions.

    :param mask:
    :return: mask with -1 on first change
    """
    change_ind = torch.argmax(mask.int(), axis=1)
    no_change_ind = torch.sum(mask, axis=1)
    change_ind[torch.where(no_change_ind == 0)[0]] = -1
    return change_ind


def calculate_errors(
    real: torch.Tensor, pred: torch.Tensor, seq_len: int
) -> Tuple[int, int, int, int, List[float], List[float]]:
    """Calculate confusion matrix, detection delay and time to false alarms.

    :param real: real labels of change points
    :param pred: predicted labels (0 or 1) of change points
    :param seq_len: length of sequence
    :return: tuple of
        - TN, FP, FN, TP
        - array of times to false alarms
        - array of detection delays
    """

    FP_delay = torch.zeros_like(real, requires_grad=False)
    delay = torch.zeros_like(real, requires_grad=False)

    tn_mask = torch.logical_and(real == pred, real == -1)
    fn_mask = torch.logical_and(real != pred, pred == -1)
    tp_mask = torch.logical_and(real <= pred, real != -1)
    fp_mask = torch.logical_or(
        torch.logical_and(torch.logical_and(real > pred, real != -1), pred != -1),
        torch.logical_and(pred != -1, real == -1),
    )

    TN = tn_mask.sum().item()
    FN = fn_mask.sum().item()
    TP = tp_mask.sum().item()
    FP = fp_mask.sum().item()

    FP_delay[tn_mask] = seq_len
    FP_delay[fn_mask] = seq_len
    FP_delay[tp_mask] = real[tp_mask]
    FP_delay[fp_mask] = pred[fp_mask]

    delay[tn_mask] = 0
    delay[fn_mask] = seq_len - real[fn_mask]
    delay[tp_mask] = pred[tp_mask] - real[tp_mask]
    delay[fp_mask] = 0

    assert (TN + TP + FN + FP) == len(real)

    return TN, FP, FN, TP, FP_delay, delay


def calculate_metrics(
    true_labels: torch.Tensor, predictions: torch.Tensor
) -> Tuple[int, int, int, int, np.array, np.array, int]:
    """Calculate confusion matrix, detection delay, time to false alarms, covering.

    :param true_labels: true labels (0 or 1) of change points
    :param predictions: predicted labels (0 or 1) of change points
    :return: tuple of
        - TN, FP, FN, TP
        - array of times to false alarms
        - array of detection delays
        - covering
    """
    mask_real = ~true_labels.eq(true_labels[:, 0][0])
    mask_predicted = ~predictions.eq(true_labels[:, 0][0])
    seq_len = true_labels.shape[1]

    real_change_ind = find_first_change(mask_real)
    predicted_change_ind = find_first_change(mask_predicted)

    TN, FP, FN, TP, FP_delay, delay = calculate_errors(
        real_change_ind, predicted_change_ind, seq_len
    )
    cover = calculate_cover(real_change_ind, predicted_change_ind, seq_len)

    return TN, FP, FN, TP, FP_delay, delay, cover


def get_models_predictions(
    inputs: torch.Tensor,
    model: nn.Module,
    model_type: str = "seq2seq",
    device: str = "cuda",
    scales: float = None,
) -> List[torch.Tensor]:
    """Get model's prediction.

    :param inputs: input data
    :param model: CPD model
    :param model_type: default "seq2seq" for BCE model, "klcpd" for KLCPD model
    :param device: device name
    :param scales: scale parameter for KL-CPD predictions
    :return: model's predictions
    """
    inputs = inputs.to(device)

    # TODO check detach
    if model_type == "klcpd":
        outs_list = klcpd.get_klcpd_output_scaled(model, inputs, model.window_1, model.window_2, scales)
    else:  # for bce model
        outs_list = [model(inputs)]
    return outs_list


def evaluate_metrics_on_set(
    model: nn.Module,
    test_loader: DataLoader,
    thresholds: Sequence[float],
    verbose: bool = True,
    model_type: str = "seq2seq",
    subseq_len: int = None,
    device: str = "cuda",
    scales: Optional[Sequence] = None,
) -> Tuple[int, int, int, int, float, float]:
    """Calculate metrics for CPD model."""
    # calculate metrics on set
    model.eval()
    model.to(device)

    n_scales, n_thresholds = len(scales), len(thresholds)
    FP_delays, delays, covers = [defaultdict(list) for _ in range(3)]
    mean_FP_delay, mean_delay, mean_cover = [
        np.zeros((n_scales, n_thresholds)) for _ in range(3)
    ]
    TN, FP, FN, TP = [
        np.zeros((n_scales, n_thresholds), dtype=np.int32) for _ in range(4)
    ]

    t_forward, t_metric = 0, 0
    with torch.no_grad():

        for test_inputs, test_labels in test_loader:
            t0 = time()
            test_labels = test_labels.to(device)
            test_out_list = get_models_predictions(
                test_inputs,
                model,
                model_type=model_type,
                device=device,
                scales=scales,
            )
            for s, (scale, test_out) in enumerate(zip(scales, test_out_list)):
                try:
                    test_out = test_out.squeeze(2)
                except ValueError:
                    try:
                        test_out = test_out.squeeze(1)
                    except ValueError:
                        test_out = test_out

                for t, threshold in enumerate(thresholds):

                    tn, fp, fn, tp, FP_delay, delay, cover = calculate_metrics(test_labels,
                                                                               test_out > threshold
                    )
                    if "cuda" in device:
                        torch.cuda.empty_cache()

                    TN[s, t] += tn
                    FP[s, t] += fp
                    FN[s, t] += fn
                    TP[s, t] += tp

                    FP_delays[(s, t)].append(FP_delay.detach().cpu())
                    delays[(s, t)].append(delay.detach().cpu())
                    covers[(s, t)].extend(cover)

                del test_out
                gc.collect()

            del test_labels
            gc.collect()

    # FIXME change `scale` to `t` and `scales` to `range(n_scales)`
    for s in range(n_scales):
        for t in range(n_thresholds):
            mean_FP_delay[s, t] = torch.cat(FP_delays[(s, t)]).float().mean().item()
            mean_delay[s, t] = torch.cat(delays[(s, t)]).float().mean().item()
            mean_cover[s, t] = np.mean(covers[(s, t)])

    if verbose:
        for s in range(n_scales):
            for t in range(n_thresholds):
                print(
                    "Scale: {}, TN: {}, FP: {}, FN: {}, TP: {}, DELAY:{}, FP_DELAY:{}, COVER: {}".format(
                        scale,
                        TN[s, t],
                        FP[s, t],
                        FN[s, t],
                        TP[s, t],
                        mean_delay[(s, t)],
                        mean_FP_delay[(s, t)],
                        mean_cover[(s, t)],
                    )
                )

    del FP_delays
    del delays
    del covers

    gc.collect()
    if "cuda" in device:
        torch.cuda.empty_cache()

    return TN, FP, FN, TP, mean_delay, mean_FP_delay, mean_cover


def area_under_graph(delay_list: List[float], fp_delay_list: List[float]) -> float:
    """Calculate area under Delay - FP delay curve.

    :param delay_list: list of delays
    :param fp_delay_list: list of fp delays
    :return: area under curve
    """
    return np.trapz(delay_list, fp_delay_list)


# ---------------------------------------------------------------------------------------
# code from TCPDBench START


def overlap(A, B):
    """Return the overlapping (i.e. Jaccard index) of two sets.

    Example of usage:
    - overlap({1, 2, 3}, set()) = 0.0
    - overlap({1, 2, 3}, {2, 5}) = 0.25
    - overlap(set(), {1, 2, 3}) = 0.0
    - overlap({1, 2, 3}, {1, 2, 3}) = 1.0
    """
    return len(A.intersection(B)) / len(A.union(B))


def partition_from_cps(locations, n_obs):
    """Return a list of sets that give a partition of the set [0, T-1], as defined by the CP locations.

    Example of usage:
    - partition_from_cps([], 5) = [{0, 1, 2, 3, 4}]
    - partition_from_cps([3, 5], 8) = [{0, 1, 2}, {3, 4}, {5, 6, 7}]
    - partition_from_cps([1,2,7], 8) = [{0}, {1}, {2, 3, 4, 5, 6}, {7}]
    - partition_from_cps([0, 4], 6) = [{0, 1, 2, 3}, {4, 5}]
    """
    partition = []
    current = set()

    all_cps = iter(sorted(set(locations)))
    cp = next(all_cps, None)
    for i in range(n_obs):
        if i == cp:
            if current:
                partition.append(current)
            current = set()
            cp = next(all_cps, None)
        current.add(i)
    partition.append(current)
    return partition


def cover_single(true_partitions, pred_partitions):
    """Compute the covering of a true segmentation by a predicted segmentation."""
    seq_len = sum(map(len, pred_partitions))
    assert seq_len == sum(map(len, true_partitions))

    cover = 0
    for t_part in true_partitions:
        cover += len(t_part) * max(
            overlap(t_part, p_part) for p_part in pred_partitions
        )
    cover /= seq_len
    return cover


def calculate_cover(
    real_change_ind: torch.Tensor, predicted_change_ind: torch.Tensor, seq_len: int
) -> List[float]:
    """Calculate covering metric.

    :param real_change_ind: indexes of real change points
    :param predicted_change_ind: indexes of predicted change points
    :param seq_len: length of considered sequence
    :return: covering value
    """
    covers = []

    for real, pred in zip(real_change_ind, predicted_change_ind):
        true_partition = partition_from_cps([real.item()], seq_len)
        pred_partition = partition_from_cps([pred.item()], seq_len)
        covers.append(cover_single(true_partition, pred_partition))

    return covers


def F1_score(confusion_matrix: Tuple[int, int, int, int]) -> float:
    """Calculate F1-score for change point detection.

    :param confusion_matrix: confusion matrix for CPD
    :return: f1 score
    """
    TN, FP, FN, TP = confusion_matrix
    f1_score = 2.0 * TP / (2 * TP + FN + FP)
    return f1_score


#########################################################################################
def evaluation_pipeline(
    model: nn.Module,
    test_dataloader: DataLoader,
    threshold_list: List[float],
    device: str = "cuda",
    verbose: bool = False,
    model_type: str = "seq2seq",
    subseq_len: int = None,
    scales: float = None,
) -> Tuple[dict, List[float], List[float]]:
    """Evaluate CPD model on test dataloader.

    :param model: CPD model
    :param test_dataloader: dataloader with test data
    :param threshold_list: list of considered threshold
    :param device: default cuda, type of device for calculation
    :param verbose: default False, if True print metrics
    :param model_type: type of evaluated model (seq2seq or klcpd)
    :param subseq_len: length of subsequence for baseline modes
    :param scales: default None, multiplier for scaling predictions
    :return:
    """
    try:
        model.to(device)
        model.eval()
    except:
        print("Cannot move model to device")

    if scales is None:
        scales = ["none"]

    n_scales, n_thresholds = len(scales), len(threshold_list)

    (
        delay_dict_2d,
        fp_delay_dict_2d,
        confusion_matrix_dict_2d,
        cover_dict_2d,
        f1_dict_2d,
    ) = [defaultdict(dict) for _ in range(5)]

    TN, FP, FN, TP, mean_delay, mean_fp_delay, cover = evaluate_metrics_on_set(
        model=model,
        test_loader=test_dataloader,
        thresholds=threshold_list,
        verbose=verbose,
        model_type=model_type,
        device=device,
        scales=scales,
    )

    for s, scale in enumerate(scales):
        for t, threshold in enumerate(threshold_list):
            confusion_matrix_dict_2d[scale][threshold] = (
                TN[s, t],
                FP[s, t],
                FN[s, t],
                TP[s, t],
            )
            delay_dict_2d[scale][threshold] = mean_delay[s, t]
            fp_delay_dict_2d[scale][threshold] = mean_fp_delay[s, t]

            cover_dict_2d[scale][threshold] = cover[s, t]
            f1_dict_2d[scale][threshold] = F1_score(
                (TN[s, t], FP[s, t], FN[s, t], TP[s, t])
            )

    best_metrics = {}
    for scale in scales:
        confusion_matrix_dict = confusion_matrix_dict_2d[scale]
        delay_dict = delay_dict_2d[scale]
        fp_delay_dict = fp_delay_dict_2d[scale]
        f1_dict = f1_dict_2d[scale]
        cover_dict = cover_dict_2d[scale]

        auc = area_under_graph(list(delay_dict.values()), list(fp_delay_dict.values()))

        # Conf matrix and F1
        best_th_f1 = max(f1_dict, key=f1_dict.get)

        best_conf_matrix = (
            confusion_matrix_dict[best_th_f1][0],
            confusion_matrix_dict[best_th_f1][1],
            confusion_matrix_dict[best_th_f1][2],
            confusion_matrix_dict[best_th_f1][3],
        )
        best_f1 = f1_dict[best_th_f1]

        # Cover
        best_cover = cover_dict[best_th_f1]

        best_th_cover = max(cover_dict, key=cover_dict.get)
        max_cover = cover_dict[best_th_cover]

        # Time to FA, detection delay
        best_time_to_FA = fp_delay_dict[best_th_f1]
        best_delay = delay_dict[best_th_f1]

        if verbose:
            print("Scale:", scale)
            print("AUC:", round(auc, 4))
            print(
                "Time to FA {}, delay detection {} for best-F1 threshold: {}".format(
                    round(best_time_to_FA, 4),
                    round(best_delay, 4),
                    round(best_th_f1, 4),
                )
            )
            print(
                "TN {}, FP {}, FN {}, TP {} for best-F1 threshold: {}".format(
                    best_conf_matrix[0],
                    best_conf_matrix[1],
                    best_conf_matrix[2],
                    best_conf_matrix[3],
                    round(best_th_f1, 4),
                )
            )
            print(
                "Max F1 {}: for best-F1 threshold {}".format(
                    round(best_f1, 4), round(best_th_f1, 4)
                )
            )
            print(
                "COVER {}: for best-F1 threshold {}".format(
                    round(best_cover, 4), round(best_th_f1, 4)
                )
            )

            print(
                "Max COVER {}: for threshold {}".format(
                    round(cover_dict[max(cover_dict, key=cover_dict.get)], 4),
                    round(max(cover_dict, key=cover_dict.get), 4),
                )
            )

        best_metrics[scale] = (
            best_th_f1,
            best_time_to_FA,
            best_delay,
            auc,
            best_conf_matrix,
            best_f1,
            best_cover,
            best_th_cover,
            max_cover,
        )

    return best_metrics, delay_dict_2d, fp_delay_dict_2d


def write_metrics_to_file(filename: str, metrics_local_dict: dict, seed: int) -> None:
    """Write result to the file.

    :param filename: path to file
    :param metrics_local_dict: dict with result metric
    :param seed: additional info for writing
    """
    with open(filename, "a") as f:
        for scale, metrics_local in metrics_local_dict.items():
            (
                best_th_f1,
                best_time_to_FA,
                best_delay,
                auc,
                best_conf_matrix,
                best_f1,
                best_cover,
                best_th_cover,
                max_cover,
            ) = metrics_local

            f.writelines("SEED: {}, Scale: {}\n".format(seed, scale))
            f.writelines("AUC: {}\n".format(auc))
            f.writelines(
                "Time to FA {}, delay detection {} for best-F1 threshold: {}\n".format(
                    round(best_time_to_FA, 4),
                    round(best_delay, 4),
                    round(best_th_f1, 4),
                )
            )
            f.writelines(
                "TN {}, FP {}, FN {}, TP {} for best-F1 threshold: {}\n".format(
                    best_conf_matrix[0],
                    best_conf_matrix[1],
                    best_conf_matrix[2],
                    best_conf_matrix[3],
                    round(best_th_f1, 4),
                )
            )
            f.writelines(
                "Max F1 {}: for best-F1 threshold {}\n".format(
                    round(best_f1, 4), round(best_th_f1, 4)
                )
            )
            f.writelines(
                "COVER {}: for best-F1 threshold {}\n".format(
                    round(best_cover, 4), round(best_th_f1, 4)
                )
            )

            f.writelines(
                "Max COVER {}: for threshold {}\n".format(max_cover, best_th_cover)
            )
            f.writelines(
                "----------------------------------------------------------------------\n"
            )
