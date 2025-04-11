# Ultralytics YOLO 🚀, AGPL-3.0 license

from .bot_sort import BOTSORT
from .byte_tracker import BYTETracker
# from trackers.sort_tracker import SortTracker
# from trackers.botsort_tracker import BotTracker
# from trackers.c_biou_tracker import C_BIoUTracker
# from trackers.ocsort_tracker import OCSortTracker
# from trackers.deepsort_tracker import DeepSortTracker
# from trackers.strongsort_tracker import StrongSortTracker
# from trackers.sparse_tracker import SparseTracker
# from trackers.ucmc_tracker import UCMCTracker
# from trackers.hybridsort_tracker import HybridSortTracker
from .track import register_tracker

from ultralytics.trackers.trackers.sort_tracker import SortTracker
from ultralytics.trackers.trackers.botsort_tracker import BotTracker
from ultralytics.trackers.trackers.c_biou_tracker import C_BIoUTracker
from ultralytics.trackers.trackers.ocsort_tracker import OCSortTracker
from ultralytics.trackers.trackers.deepsort_tracker import DeepSortTracker
from ultralytics.trackers.trackers.strongsort_tracker import StrongSortTracker
from ultralytics.trackers.trackers.sparse_tracker import SparseTracker
from ultralytics.trackers.trackers.ucmc_tracker import UCMCTracker
from ultralytics.trackers.trackers.hybridsort_tracker import HybridSortTracker

__all__ = ('register_tracker', 'BOTSORT', 'BYTETracker', 'SortTracker', 'BotTracker', 'C_BIoUTracker',
           'OCSortTracker', 'DeepSortTracker', 'StrongSortTracker', 'SparseTracker', 'UCMCTracker',
           'HybridSortTracker')  # allow simpler import

# __all__ = ('register_tracker', 'BOTSORT', 'BYTETracker')  # allow simpler import