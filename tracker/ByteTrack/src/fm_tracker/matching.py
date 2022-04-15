import cv2
from matplotlib.pyplot import bar
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

from cython_bbox import bbox_overlaps as bbox_ious
from .tracking_utils import kalman_filter
import time

def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def minarea_ious(atlwhs, btlwhs):
    """
    Compute cost based on IoU
    :type atlwhs: list[tlwh] | np.ndarray
    :type btlwhs: list[tlwh] | np.ndarray

    :rtype ious np.ndarray
    """
    _minarea_iou_ = np.zeros((len(atlwhs),len(btlwhs)), dtype=np.float)
    if _minarea_iou_.size == 0:
        return _minarea_iou_

    atlwhs_np = np.ascontiguousarray(atlwhs, dtype=np.float)
    btlwhs_np = np.ascontiguousarray(btlwhs, dtype=np.float)
    aarea_square = atlwhs_np[:, 2] * atlwhs_np[:, 3]
    barea_square = btlwhs_np[:, 2] * btlwhs_np[:, 3]

    for k in range(atlwhs_np.shape[0]):
        for n in range(btlwhs_np.shape[0]):
            minarea = min(aarea_square[k], barea_square[n])
            iw = (
                min(btlwhs_np[n, 0] + btlwhs_np[n, 2],  atlwhs_np[k, 0] + atlwhs_np[k, 2]) -
                max(btlwhs_np[n, 0], atlwhs_np[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(btlwhs_np[n, 1] + btlwhs_np[n, 3], atlwhs_np[k, 1] + atlwhs_np[k, 3]) -
                    max(btlwhs_np[n, 1], atlwhs_np[k, 1]) + 1
                )
                if ih > 0:
                    _minarea_iou_[k, n] = iw * ih / minarea
    return _minarea_iou_


def minarea_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlwhs = atracks
        btlwhs = btracks
    else:
        atlwhs = [track.tlwh for track in atracks]
        btlwhs = [track.tlwh for track in btracks]
    _mious = minarea_ious(atlwhs, btlwhs)
    cost_matrix = 1 - _mious

    return cost_matrix


def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def pixel_d(atlbr, btlbr):
    # axis-aligned bounding box
    aAABB = [[atlbr[0], atlbr[1]], [atlbr[0], atlbr[3]], [atlbr[2], atlbr[1]], [atlbr[2], atlbr[3]]] 
    bAABB = [[btlbr[0], btlbr[1]], [btlbr[0], btlbr[3]], [btlbr[2], btlbr[1]], [btlbr[2], btlbr[3]]] 
    aAABB = np.array(aAABB)
    bAABB = np.array(bAABB)
    return np.mean(np.sqrt(np.sum(np.square(bAABB-aAABB), axis=1)))

def pixel_distance(atracks, btracks):
    """
    Compute cost based on mean of 4 tlbr points
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    cost_matrix = np.zeros((len(atracks), len(btracks)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    # small distance means similar pairs.
    cost_matrix =np.abs(cdist(atlbrs, btlbrs, pixel_d))  # Nomalized features by ceiling to 0

    return cost_matrix

def dious(atlbrs, btlbrs):
    """
    Compute cost based on DIoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype dious np.ndarray
    """
    _diou_ = np.zeros((len(atlbrs),len(btlbrs)), dtype=np.float)
    if _diou_.size == 0:
        return _diou_

    iou = ious(atlbrs, btlbrs)

    atlbrs_np = np.ascontiguousarray(atlbrs, dtype=np.float)
    btlbrs_np = np.ascontiguousarray(btlbrs, dtype=np.float)

    acenterx_np = atlbrs_np[:, 0] + (atlbrs_np[:, 2] - atlbrs_np[:, 0]) * 0.5
    acentery_np = atlbrs_np[:, 1] + (atlbrs_np[:, 3] - atlbrs_np[:, 1]) * 0.5
    bcenterx_np = btlbrs_np[:, 0] + (btlbrs_np[:, 2] - btlbrs_np[:, 0]) * 0.5
    bcentery_np = btlbrs_np[:, 1] + (btlbrs_np[:, 3] - btlbrs_np[:, 1]) * 0.5
    
    acenter_np = np.hstack((
        np.expand_dims(acenterx_np, axis=1), np.expand_dims(acentery_np, axis=1)))
    bcenter_np = np.hstack((
        np.expand_dims(bcenterx_np, axis=1), np.expand_dims(bcentery_np, axis=1)))

    for k in range(atlbrs_np.shape[0]):
        for n in range(btlbrs_np.shape[0]):
            cxd = acenter_np[k, 0] - bcenter_np[n, 0]
            cyd = acenter_np[k, 1] - bcenter_np[n, 1]
            ow = max(atlbrs_np[k, 2], btlbrs_np[n, 2]) - min(atlbrs_np[k, 0], btlbrs_np[n, 0])
            oh = max(atlbrs_np[k, 3], btlbrs_np[n, 3]) - min(atlbrs_np[k, 1], btlbrs_np[n, 1])

            c_dist = cxd**2 + cyd**2
            diag_dist = ow**2 + oh**2

            _diou_[k, n] = iou[k, n] - c_dist / diag_dist
    return _diou_


def diou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """
    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _dious = dious(atlbrs, btlbrs)
    cost_matrix = 1 - _dious
    
    return cost_matrix


def embedding_distance(tracks, detections, metric='cosine', use_last_feat=False):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    if use_last_feat:
        track_features = np.asarray([track.features[-1] for track in tracks], dtype=np.float)
    else:
        track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    return cost_matrix
    

def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    #fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_embed_score(cost_matrix, sim_matrix, detections, alpha=0.8):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * (alpha * det_scores + (1 - alpha) * sim_matrix)
    fuse_cost = 1 - fuse_sim
    return fuse_cost

