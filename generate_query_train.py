from pathlib import Path

from PIL import Image

import scipy.io as sio
import numpy as np
import math
import random

random.seed(42) # set the seed for repeatibility

# utils function to truncate coordinates to 4 digit precision
def truncate(n, digits = 4):
    str_n = str(n)
    dot_idx = str_n.find(".")
    return float(str_n[:dot_idx + digits + 1])


frame_train = sio.loadmat('dataset/frame_train.mat')
frame_test = sio.loadmat('dataset/frame_test.mat')

id_train = sio.loadmat("dataset/ID_train.mat")
id_test = sio.loadmat("dataset/ID_test.mat")


ids, xs, ys, ws, hs, fs, lines = [],[],[],[],[],[],[] 

# getting all the training crops of known people (id != -2)
for frame in frame_train.get("img_index_train"):
    f_train = frame.item()[0]
    annotation = sio.loadmat(f'dataset/annotations/{f_train}.jpg.mat')
    bbs = annotation.get("box_new", 
                        annotation.get("anno_file", 
                                       annotation.get("anno_previous", None)))
    for bb in bbs:
        if bb[0] > 0: # if query id
            id = int(bb[0])
            x1 = truncate(bb[1])
            y1 = truncate(bb[2])
            x2 = truncate(bb[3])
            y2 = truncate(bb[4])
            line = f"{id} {x1} {y1} {x2} {y2} {f_train}"

            ids.append(id)
            xs.append(x1)
            ys.append(y1)
            ws.append(x2)
            hs.append(y2)
            fs.append(f_train)
            lines.append(line)
            
            # print(line)


ids, lines = map(list, zip(*sorted(zip(ids, lines), key=lambda x: x[0]))) # ordering for id (not needed but why not)
unique_ids = np.unique(ids)

iter_ids = iter(unique_ids) # to get next id
next(iter_ids) # it must be one step ahead

n_query_per_id = 3
sampled_lines = []

# sample only 3 queries per id to create a reasonable training set
for id in unique_ids:
    first_idx = ids.index(id)
    try:
        last_idx = ids.index(next(iter_ids))
    except StopIteration:
        last_idx = ids.index(ids[-1])

    sampled_line_count = 0
    previous_sampled_line_idx = None
    while sampled_line_count < n_query_per_id:
        sampled_line_idx = random.randint(first_idx, last_idx)
        if sampled_line_idx != previous_sampled_line_idx:
            sampled_line = lines[sampled_line_idx]
            sampled_lines.append(sampled_line)
            print(sampled_line)
        previous_sampled_line_idx = sampled_line_idx
        sampled_line_count += 1

# query file name to save sampled lines
query_info_train_file = "dataset/query_info_train.txt"

# write lines to a file
with open(query_info_train_file, "w") as f:
    f.write("\n".join(sampled_lines))

# directories
img_dir = Path("dataset/frames")
write_dir = Path("dataset/query_box_train")
write_dir.mkdir(exist_ok=True)

# read annotation file
with open(query_info_train_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        
        parts = line.strip().split()

        # parse fields
        ID = int(parts[0])

        x = float(parts[1])
        y = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])

        img_name = parts[5]

        # load image
        img_path = img_dir / f"{img_name}.jpg"
        img = Image.open(img_path)

        # PIL crop uses:
        # (left, upper, right, lower)
        crop_box = (
            int(round(x - 0.5)),
            int(round(y - 0.5)),
            int(round(x + w - 0.5)),
            int(round(y + h - 0.5)),
)

        cropped = img.crop(crop_box)

        # output filename
        output_name = f"{ID:03d}_{img_name}.jpg"
        output_path = write_dir / output_name

        # save cropped image
        cropped.save(output_path)

        print(f"Saved: {output_path}")