KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Maps bones to a matplotlib color name.
# Maps bones to a unique matplotlib color name.
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'red',
    (0, 2): 'blue',
    (1, 3): 'green',
    (2, 4): 'purple',
    (0, 5): 'orange',
    (0, 6): 'pink',
    (5, 7): 'brown',
    (7, 9): 'olive',
    (6, 8): 'cyan',
    (8, 10): 'magenta',
    (5, 6): 'yellow',
    (5, 11): 'lime',
    (6, 12): 'teal',
    (11, 12): 'indigo',
    (11, 13): 'gold',
    (13, 15): 'plum',
    (12, 14): 'coral',
    (14, 16): 'darkgreen'
}

SAMPLING_RATE = 10
CONF_THRESHOLD = 0.5