from matplotlib.pyplot import set_cmap
import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist

from .tracking_utils.kalman_filter import KalmanFilter
from . import matching
from .basetrack import BaseTrack, TrackState
from .zone import zone

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, temp_feat, cid, buffer_size=30):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.features = deque([], maxlen=buffer_size)
        self.update_features(temp_feat)
        self.alpha = 0.9

        # record original detection box.
        self.det_tlwh = np.asarray(tlwh, dtype=np.float)

        # zone
        self.cam_id = cid
        self.zone = zone()
        self.set_cam(cid)
        self.zone_area = self.get_zone()

        # occlusion object
        self.occlusion_tracks = {}

    def add_occlusion_track(self, track):
        if track.track_id not in self.occlusion_tracks:
            self.occlusion_tracks.setdefault(track.track_id, track)
        else:
            self.occlusion_tracks[track.track_id] = track

    def remove_occlusion_tracks(self, track_id):
        if track_id in self.occlusion_tracks:
            del self.occlusion_tracks[track_id]

    def set_cam(self, cid):
        if isinstance(cid, str):
            cid = int(cid.split("c")[-1])
        self.zone.set_cam(cid)
    
    def get_zone(self):
        return self.zone.get_zone(self.det_tlwh)

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 0:
            self.is_activated = True
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.det_tlwh = new_track.det_tlwh
        self.zone_area = self.get_zone()

    def recall(self, kalman_filter, old_track, frame_id):
        # Only keep old track id and features
        self.track_id = old_track.track_id
        self.kalman_filter = kalman_filter
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

        new_current_feat = self.curr_feat
        self.features = copy.deepcopy(old_track.features)
        self.smooth_feat = copy.deepcopy(old_track.smooth_feat)
        self.update_features(new_current_feat)

    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.det_tlwh = new_track.det_tlwh
        self.zone_area = self.get_zone()
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)
    
    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, track_thresh, match_thresh, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.with_occlusion_stracks = []   # type: list[STrack] 
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.match_thresh = match_thresh
        self.track_thresh = track_thresh
        self.low_det_thresh = track_thresh
        self.high_det_thresh = track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * 30)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    def update(self, output_results, id_feature, cid, use_embedding=False):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        with_occlusion_stracks = []
        lost_stracks = []
        removed_stracks = []
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]

        inds_low = scores > 0.1
        inds_high = scores < self.track_thresh

        remain_inds = scores > self.track_thresh
        inds_second = np.logical_and(inds_low, inds_high)
       
        dets_high = output_results[remain_inds]
        dets_low = output_results[inds_second]

        # print(f"1st idx num: {remain_inds.shape}")
        # print(f"2nd idx num: {inds_second.shape}")

        if len(dets_high) > 0:
            '''Detections'''
            remain_id_feature = id_feature[remain_inds]
            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, cid, 30) for
                        (tlbrs, f) in zip(dets_high[:,:5], remain_id_feature)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        
        # calculate similarity matrix of trackers and embedding
        if len(strack_pool) > 0 and use_embedding: 
            track_features = np.asarray([track.smooth_feat for track in strack_pool], dtype=np.float)
            sim_matrix = np.maximum(0.0, cdist(track_features, id_feature, 'cosine'))  # Nomalized features
        else:
            sim_matrix = None
        
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        if use_embedding and sim_matrix is not None:
            sim_matrix = sim_matrix[:, remain_inds]
            assert sim_matrix.shape == dists.shape
            dists = matching.fuse_embed_score(dists, sim_matrix, detections, alpha=0.9)
        else:
            dists = matching.fuse_score(dists, detections)

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id, update_feature=True)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)            
            for wostrack in self.with_occlusion_stracks:
                if track.track_id in wostrack.occlusion_tracks:
                    wostrack.remove_occlusion_tracks(track.track_id)


        ''' Step 3: Second association, with low score detection boxes (Occlusion Object)'''
        # association the untrack to the low score detections
        if len(dets_low) > 0:
            '''Detections'''
            second_id_feature = id_feature[inds_second]
            detections_low = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, cid, 30) for
                        (tlbrs, f) in zip(dets_low[:, :5], second_id_feature)]
        else:
            detections_low = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        # dists = matching.pixel_distance(r_tracked_stracks, detections_low)
        # matches, u_track, u_detection_low = matching.linear_assignment(dists, thresh=150)
        dists = matching.iou_distance(r_tracked_stracks, detections_low)
        matches, u_track, u_detection_low = matching.linear_assignment(dists, thresh=0.5)
        # dists = matching.diou_distance(r_tracked_stracks, detections_second)
        # matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            
            det = detections_low[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, update_feature=True)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
            for wostrack in self.with_occlusion_stracks:
                if track.track_id in wostrack.occlusion_tracks:
                    wostrack.remove_occlusion_tracks(track.track_id)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                if (track.get_zone() == 3 or track.get_zone() == 4) and len(activated_starcks) > 0:
                    # find the nearest tracker
                    dist = 1 - matching.minarea_iou_distance([track], activated_starcks)
                    nearest_idx = np.argmax(dist)
                    if dist[:, nearest_idx] > 0.8:
                        nearest_track = activated_starcks[nearest_idx]
                        nearest_track.add_occlusion_track(track)
                        with_occlusion_stracks.append(nearest_track)

                track.mark_lost()
                lost_stracks.append(track)
        
        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        """u_detection_second & u_detection"""
        detections = [detections[i] for i in u_detection]
        detections += [detections_low[i] for i in u_detection_low]
        """Only u_detection"""
        # detections = [detections[i] for i in u_detection]
        # dists = matching.diou_distance(unconfirmed, detections)
        dists = matching.iou_distance(unconfirmed, detections)
        dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id, update_feature=True)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.low_det_thresh:
                continue
            
            octrack_candidates = []
            woctrack_idx = []
            for i in range(len(self.with_occlusion_stracks)):
                wostrack = self.with_occlusion_stracks[i]
                for tid, occlusion_track in wostrack.occlusion_tracks.items():
                    octrack_candidates.append(occlusion_track)
                    woctrack_idx.append(i)
            if len(octrack_candidates) > 0:
                iou_dist = 1 - matching.minarea_iou_distance(
                    [track], 
                    [self.with_occlusion_stracks[j] for j in woctrack_idx]
                )
                embed_dist = matching.embedding_distance([track], octrack_candidates, metric='cosine')
                dist = iou_dist * (0.5 * embed_dist + 0.5 * track.score)
                nearest_idx = np.argmax(dist)
                if dist[:,nearest_idx] > 0.4 and track.score >= self.high_det_thresh:
                    track.recall(self.kalman_filter, octrack_candidates[nearest_idx], self.frame_id)
                    print(f"recall id: {track.track_id} - {track.cam_id}")
                    self.with_occlusion_stracks[i].remove_occlusion_tracks(track.track_id)
                else:
                    track.activate(self.kalman_filter, self.frame_id)
            else:
                track.activate(self.kalman_filter, self.frame_id)
            
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Step 6: Update occlusions"""
        wait_for_del_ost = []
        for track in self.with_occlusion_stracks:
            if len(track.occlusion_tracks) == 0:
                wait_for_del_ost.append(track)
                continue
            del_tids = [] 
            for tid, occlusion_track in track.occlusion_tracks.items():
                if self.frame_id - occlusion_track.end_frame > 30:
                    del_tids.append(tid)
            for tid in del_tids:
                track.remove_occlusion_tracks(tid)
        self.with_occlusion_stracks = sub_stracks(self.with_occlusion_stracks, wait_for_del_ost)
        self.with_occlusion_stracks.extend(with_occlusion_stracks)


        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
