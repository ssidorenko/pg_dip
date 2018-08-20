import math

import numpy as np


def get_regions(img, v_division, h_division, overlap=10):
    size_v = math.ceil(img.shape[1] / v_division)
    size_h = math.ceil(img.shape[2] / h_division)

    out = []

    for i_v in range(v_division):
        row = []
        for i_h in range(h_division):
            v_start = max(0, i_v * size_v - overlap)
            v_end = min(img.shape[1], (i_v + 1) * size_v + overlap)
            h_start = max(0, i_h * size_h - overlap)
            h_end = min(img.shape[2], (i_h + 1) * size_h + overlap)
            sliced = img[:, v_start:v_end, h_start:h_end]

            # Reflection padding if we're the edge of the image
            if i_v == 0:
                padding = sliced[:, overlap-1::-1, :]
                sliced = np.concatenate([padding, sliced], axis=1)
            elif i_v == v_division - 1:
                padding = sliced[:, img.shape[1]:-overlap-1:-1, :]
                sliced = np.concatenate([sliced, padding], axis=1)

            if i_h == 0:
                padding = sliced[:, :, overlap-1::-1]
                sliced = np.concatenate([padding, sliced], axis=2)
            elif i_h == h_division - 1:
                padding = sliced[:, :, img.shape[2]:-overlap-1:-1]
                sliced = np.concatenate([sliced, padding], axis=2)

            row.append(sliced)
        out.append(row)
    return out


def image_from_regions(regions, overlap=10):
    region_v_size = regions[0][0].shape[1] - overlap * 2
    region_h_size = regions[0][0].shape[2] - overlap * 2

    out_v_size = region_v_size * len(regions)
    out_h_size = region_h_size * len(regions[0])

    out = np.empty((3, out_v_size, out_h_size), dtype=regions[0][0].dtype)

    # First, simply stitch together the separate regions minus the overlap region

    for i_v, row in enumerate(regions):
        for i_h, region in enumerate(row):
            out[:, region_v_size * i_v:region_v_size * (i_v+1), region_h_size * i_h:region_h_size * (i_h+1)] = region[:, overlap:-overlap, overlap:-overlap]

    # Then, apply crossfading on vertical borders
    v_fade_weights = np.meshgrid(np.ones(region_v_size), np.linspace(0.0, 1.0, num=overlap*2), indexing='ij')[1]
    for i_v, row in enumerate(regions):
        for i_h, region in enumerate(row):
            if i_h != len(row) - 1:

                out[
                    :,
                    region_v_size * i_v:region_v_size * (i_v+1),
                    region_h_size * (i_h+1) - overlap: region_h_size * (i_h+1) + overlap
                ] = (1 - v_fade_weights) * regions[i_v][i_h][:, overlap: -overlap, -overlap * 2:] + \
                    v_fade_weights * regions[i_v][i_h + 1][:, overlap: -overlap, :overlap * 2]

    # Then, apply crossfading on horizontal borders
    h_fade_weights = np.meshgrid(np.linspace(0.0, 1.0, num=overlap*2), np.ones(region_h_size),  indexing='ij')[0]
    for i_v, row in enumerate(regions):
        if i_v != len(regions) -1:
            for i_h, region in enumerate(row):
                out[
                        :,
                        region_v_size * (i_v + 1) - overlap:region_v_size * (i_v + 1) + overlap,
                        region_h_size * i_h: region_h_size * (i_h+1)
                    ] = (1 - h_fade_weights) * regions[i_v][i_h][:, -overlap * 2:, overlap:-overlap] + \
                    h_fade_weights * regions[i_v+1][i_h][:, :overlap * 2, overlap:-overlap]

    # Then, crossfade on intersections
    v_fade_weights = np.meshgrid(np.ones(overlap*2), np.linspace(0.0, 1.0, num=overlap*2), indexing='ij')[1]
    h_fade_weights = np.meshgrid(np.linspace(0.0, 1.0, num=overlap*2), np.ones(overlap*2),  indexing='ij')[0]

    for i_v, row in enumerate(regions):
        if i_v != len(regions) -1:
            for i_h, region in enumerate(row):
                if i_h != len(row) - 1:
                    out[
                        :,
                        region_v_size * (i_v + 1) - overlap:region_v_size * (i_v + 1) + overlap,
                        region_h_size * (i_h+1) - overlap: region_h_size * (i_h+1) + overlap
                    ] = regions[i_v][i_h][:, -overlap *2:, -overlap*2:] * (1-v_fade_weights) * (1-h_fade_weights) + \
                        regions[i_v][i_h + 1][:, -overlap *2:, :overlap*2] * v_fade_weights * (1 - h_fade_weights) + \
                        regions[i_v + 1][i_h][:, :overlap *2, -overlap*2:] * (1 - v_fade_weights) * h_fade_weights + \
                        regions[i_v + 1][i_h + 1][:, :overlap *2, :overlap*2] * v_fade_weights * h_fade_weights

    return out
