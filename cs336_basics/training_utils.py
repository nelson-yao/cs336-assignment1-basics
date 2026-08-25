import math


def lr_schedule(t, amax, amin, tw, tc):
    if t < tw:
        return t / tw * amax
    elif t < tc:
        return amin + 0.5 * (1 + math.cos((t - tw) / (tc - tw) * math.pi)) * (
            amax - amin
        )
    else:
        return amin
