import numpy as np

def dot(a, b):
    return sum(a[i] * b[i] for i in range(len(a)))

def mag(v):
    return np.sqrt(sum(el**2 for el in v))

def get_angle(a, b):
    cos_theta = dot(a, b) / (mag(a) * mag(b))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Prevent floating-point errors
    return np.arccos(cos_theta)  # Returns angle in radians