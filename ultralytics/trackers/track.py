# Ultralytics YOLO 🚀, AGPL-3.0 license

from functools import partial

import torch

from ultralytics.utils import IterableSimpleNamespace, yaml_load
from ultralytics.utils.checks import check_yaml

from .bot_sort import BOTSORT
from .byte_tracker import BYTETracker

# from trackers.byte_tracker import ByteTracker
# from trackers.sort_tracker import SortTracker
# from trackers.botsort_tracker import BotTracker
# from trackers.c_biou_tracker import C_BIoUTracker
# from trackers.ocsort_tracker import OCSortTracker
# from trackers.deepsort_tracker import DeepSortTracker
# from trackers.strongsort_tracker import StrongSortTracker
# from trackers.sparse_tracker import SparseTracker
# from trackers.ucmc_tracker import UCMCTracker
# from trackers.hybridsort_tracker import HybridSortTracker

from ultralytics.trackers.trackers.sort_tracker import SortTracker
from ultralytics.trackers.trackers.botsort_tracker import BotTracker
from ultralytics.trackers.trackers.c_biou_tracker import C_BIoUTracker
from ultralytics.trackers.trackers.ocsort_tracker import OCSortTracker
from ultralytics.trackers.trackers.deepsort_tracker import DeepSortTracker
from ultralytics.trackers.trackers.strongsort_tracker import StrongSortTracker
from ultralytics.trackers.trackers.sparse_tracker import SparseTracker
from ultralytics.trackers.trackers.ucmc_tracker import UCMCTracker
from ultralytics.trackers.trackers.hybridsort_tracker import HybridSortTracker

TRACKER_MAP = {'bytetrack': BYTETracker, 'botsort': BOTSORT, 'sort': SortTracker, 'botsort': BotTracker,
               'c_bioutrack': C_BIoUTracker, 'ocsort': OCSortTracker, 'deepsort': DeepSortTracker,
               'strongsort': StrongSortTracker, 'sparsetrack': SparseTracker, 'ucmctrack': UCMCTracker,
               'hybridsort': HybridSortTracker}

# TRACKER_MAP = {'bytetrack': BYTETracker, 'botsort': BOTSORT}

def on_predict_start(predictor, persist=False):
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.

    Raises:
        AssertionError: If the tracker_type is not 'bytetrack' or 'botsort'.
    """
    if hasattr(predictor, 'trackers') and persist:
        return
    tracker = check_yaml(predictor.args.tracker)
    cfg = IterableSimpleNamespace(**yaml_load(tracker))
    assert cfg.tracker_type in ['bytetrack', 'botsort', 'sort', 'botsort', 'c_bioutrack', 'ocsort', 'deepsort',
                                'strongsort', 'sparsetrack', 'ucmctrack', 'hybridsort'], \
        f"Only support 'bytetrack' and 'botsort' for now, but got '{cfg.tracker_type}'"
    trackers = []
    for _ in range(predictor.dataset.bs):
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)
        trackers.append(tracker)
    predictor.trackers = trackers


def on_predict_postprocess_end(predictor):
    """Postprocess detected boxes and update with object tracking."""
    bs = predictor.dataset.bs
    im0s = predictor.batch[1]
    for i in range(bs):
        det = predictor.results[i].boxes.cpu().numpy()
        if len(det) == 0:
            continue
        # tracks = predictor.trackers[i].update(det, im0s[i])
        tracks, bias = predictor.trackers[i].update(det, im0s[i])
        predictor.count_bias = bias
        if len(tracks) == 0:
            continue
        idx = tracks[:, -1].astype(int)
        predictor.results[i] = predictor.results[i][idx]
        predictor.results[i].update(boxes=torch.as_tensor(tracks[:, :-1]))


def register_tracker(model, persist):
    """
    Register tracking callbacks to the model for object tracking during prediction.

    Args:
        model (object): The model object to register tracking callbacks for.
        persist (bool): Whether to persist the trackers if they already exist.
    """
    model.add_callback('on_predict_start', partial(on_predict_start, persist=persist))
    model.add_callback('on_predict_postprocess_end', on_predict_postprocess_end)