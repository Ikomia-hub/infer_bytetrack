import numpy as np
from onemetric.cv.utils.iou import box_iou_batch


# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks):
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)


# matches our bounding boxes with predictions
def match_detections_with_tracks(detections, tracks):
    detection_boxes = np.array([xywh_xyxy(o.box) for o in detections])
    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detection_boxes)
    track2detection = np.argmax(iou, axis=1)
    det_id_2_track_id = {}
    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            det_id_2_track_id[detection_index] = tracks[tracker_index].track_id
    return det_id_2_track_id


def xywh_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]