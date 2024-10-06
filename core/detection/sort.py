import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

class KalmanBoxTracker:
    """
    This class represents the internal state of individual tracked objects observed as bounding boxes.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initializes a tracker using the initial bounding box.
        """
        self.kf = KalmanFilter(dim_x=7, dim_z=4)  # 7 state variables, 4 measurements
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0]])
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] = 0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        # Return bounding box as a 4x1 array
        predicted_bbox = self.convert_x_to_bbox(self.kf.x)
        return predicted_bbox.reshape((4, 1))  # Ensure 4x1 shape

    def get_state(self):
        """
        Returns the current bounding box estimate in 4x1 form.
        """
        return self.convert_x_to_bbox(self.kf.x).reshape((4, 1))  # Ensure 4x1 shape

    @staticmethod
    def convert_bbox_to_z(bbox):
        """
        Converts a bounding box in the form [x1, y1, x2, y2] to z form [x, y, s, r].
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.
        y = bbox[1] + h / 2.
        s = w * h  # scale is area
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))

    @staticmethod
    def convert_x_to_bbox(x):
        """
        Converts the state vector x into a bounding box [x1, y1, x2, y2].
        """
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.])


class Sort:
    def __init__(self, max_age=40, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
        dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns the list of track objects updated for this frame in the format [x1, y1, x2, y2, obj_id].
        """
        self.frame_count += 1

        # Initialize tracker bounding boxes as a 2D array with 5 columns (x1, y1, x2, y2, ID)
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []

        for t, trk in enumerate(self.trackers):
            pos = trk.predict()  # This returns the predicted bounding box as a 4x1 array
            pos = pos.flatten()  # Flatten to a 1D array of shape (4,)
            if isinstance(pos, np.ndarray) and pos.shape == (4,):
                trks[t, :] = [pos[0], pos[1], pos[2], pos[3], trk.id]  # Include the tracker ID
            else:
                print(f"Warning: Invalid bounding box shape for tracker {t}, pos: {pos}")
                to_del.append(t)  # Mark for deletion

        # Remove invalid trackers
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # Match detections to trackers
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets, trks)

        # Update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)

        # Output tracked objects (including bbox and ID)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state().flatten()  # Get the bounding box [x1, y1, x2, y2] as a flattened 1D array
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # Append bbox and ID
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)  # Return a list of tracked bounding boxes with IDs
        return np.empty((0, 5))  # Return empty array if no tracks

    @staticmethod
    def iou(bb_test, bb_gt):
        """
        Computes IOU between two bounding boxes in the form [x1, y1, x2, y2].
        """
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
        return o

    def associate_detections_to_trackers(self, detections, trackers):
        """
        Assigns detections to tracked object (both represented as bounding boxes)
        Returns:
        matched_indices: list of matches
        unmatched_detections: list of unmatched detections
        unmatched_trackers: list of unmatched trackers
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)

        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self.iou(det, trk)

        matched_indices = linear_sum_assignment(-iou_matrix)
        matched_indices = np.array(matched_indices).T

        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)

        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

# Example usage:
# sort_tracker = Sort()
# detections = np.array([[100, 100, 200, 200, 0.9], [150, 150, 250, 250, 0.8]])  # [x1, y1, x2, y2, score]
# tracks = sort_tracker.update(detections)
# print(tracks)  # Returns tracked objects
