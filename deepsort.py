import numpy as np

class KalmanFilter:
    def __init__(self):
        ndim, dt = 4, 1.
        self._motion_mat = np.eye(2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance
    def predict(self, mean, covariance):
        mean = np.dot(self._motion_mat, mean)
        covariance = np.dot(self._motion_mat, np.dot(covariance, self._motion_mat.T))
        return mean, covariance
    def update(self, mean, covariance, measurement):
        projected_mean = np.dot(self._update_mat, mean)
        innovation = measurement - projected_mean
        mean = mean + np.dot(self._update_mat.T, innovation)
        return mean, covariance

def iou(bbox, candidates):
    x1, y1, w1, h1 = bbox
    x2, y2, w2, h2 = candidates.T

    xx1 = np.maximum(x1, x2)
    yy1 = np.maximum(y1, y2)
    xx2 = np.minimum(x1 + w1, x2 + w2)
    yy2 = np.minimum(y1 + h1, y2 + h2)

    inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union

class Detection:
    def __init__(self, tlwh, confidence):
        self.tlwh = tlwh
        self.confidence = confidence

    def to_xyah(self):
        x, y, w, h = self.tlwh
        return [x + w/2, y + h/2, w, h]

  class Track:
    def __init__(self, mean, covariance, track_id):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.time_since_update = 0


class Tracker:
    def __init__(self):
        self.kf = KalmanFilter()
        self.tracks = []
        self.next_id = 1

    def update(self, detections):
        for track in self.tracks:
            track.mean, track.covariance = self.kf.predict(track.mean, track.covariance)
            track.time_since_update += 1

        matches = []
        used_tracks = set()
        used_dets = set()

        for t, track in enumerate(self.tracks):
            best_iou = 0
            best_d = None
            for d, det in enumerate(detections):
                if d in used_dets:
                    continue

                det_box = np.array(det.tlwh, dtype=float)
                track_box = track.mean[:4]
                iou_score = iou(track_box, det_box)
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_d = d
            if best_iou > 0.3:
                matches.append((t, best_d))
                used_tracks.add(t)
                used_dets.add(best_d)
        unmatched_tracks = [t for t in range(len(self.tracks)) if t not in used_tracks]
        unmatched_dets = [d for d in range(len(detections)) if d not in used_dets]

        for t, d in matches:
            det = detections[d]
            measurement = det.to_xyah()

            self.tracks[t].mean, self.tracks[t].covariance = self.kf.update(
                self.tracks[t].mean,
                self.tracks[t].covariance,
                measurement
            )
            self.tracks[t].hits += 1
            self.tracks[t].time_since_update = 0

        for d in unmatched_dets:
            det = detections[d]
            mean, cov = self.kf.initiate(det.to_xyah())
            self.tracks.append(Track(mean, cov, self.next_id))
            self.next_id += 1

        self.tracks = [t for t in self.tracks if t.time_since_update < 30]
