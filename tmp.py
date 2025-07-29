import numpy as np

if __name__ == "__main__":
    with open("data/dataset/train/labels/1-2_000000.txt", "r") as f:
        for id, line in enumerate(f.readlines()):
            line = line.strip().split(' ')
            print(type(line))
            print(line)
            print(type(line[2:6]))
            print(line[2:6])
            print(type(line[2]))
            box = line[2:6]
            box = np.array(box, dtype=np.int32)
            bbox = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
            print(bbox)
            print(bbox[0])
            break