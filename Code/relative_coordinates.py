import numpy as np

def row_grouping(coco_annotations, threshold_value):
    ''' This function groups bounding boxes in appropriate rows
    returns the row value for each bounding box along with annotations '''

    coco_annotations = coco_annotations.tolist()
    max_height = np.max([_[3] for _ in coco_annotations])
    threshold = max_height * threshold_value
    rows = {}

    coco_annotations.sort(key=lambda r: [int(round(float(r[1]) / threshold)), r[0]])
    
    for idx, box in enumerate(coco_annotations):
        row = [int(round(float(box[1]) / threshold)), box[0]]
        box.append(row[0])
        if box[4] not in rows:
            rows[box[4]] = len(rows)
        box[4] = rows[box[4]]    
    return coco_annotations

def count_row_objects(annotations):
    ''' Count the objects frequencies in each row and append object counts'''
    ptr = annotations[0][4]
    count = 0
    annotations[0].append(count)
    for box in annotations[1:]:
        if box[4]!=ptr:
            count = 0
            box.append(count)
            ptr = box[4]
        else:
            count+=1
            box.append(count)
    return annotations