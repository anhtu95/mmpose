import numpy as np
import cv2
import os


def get_points_of_foot(list_points):
    max_y = [-1, -1]
    foot_point_x = [-1, -1]
    foot_point_y = [-1, -1]
    for p in list_points:
        if p[1] > max_y[0]:
            max_y[1] = max_y[0]
            max_y[0] = p[1]
            foot_point_x[1] = foot_point_x[0]
            foot_point_x[0] = p[0]
            foot_point_y[1] = foot_point_y[0]
            foot_point_y[0] = p[1]
        elif p[1] > max_y[1]:
            max_y[1] = p[1]
            foot_point_x[1] = p[0]
            foot_point_y[1] = p[1]
    return (foot_point_x[0], foot_point_y[0]), (foot_point_x[1], foot_point_y[1])


pts_src = np.array([
    [471, 82],
    [21, 251],
    [1097, 388],
    [2127, 227],
    [1669, 70],
    [1077, 102]
])

pts_dst = np.array([
    [6, 5],
    [6, 355],
    [319, 355],
    [632, 355],
    [632, 5],
    [319, 5]
])
out_video_root = "./output"
img_input = cv2.imread("/home/anhtu/Project/trento/computer-vision/mmpose/resources/basket_field.jpg")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videoWriter = cv2.VideoWriter(os.path.join(out_video_root, "projection_cropped.mp4"), fourcc, 25.0,
                              (img_input.shape[1], img_input.shape[0]))
homography, status = cv2.findHomography(pts_src, pts_dst)
with open("./output/keypoints_cropped.txt", 'r') as f:
    prev_frame = 0
    prev_person = 0
    key_points = []
    img = img_input.copy()
    for line in f:
        info = line.split(" ")
        frame_id, person_id, x, y = int(info[0]), int(info[1]), int(info[2]), int(info[3])
        if frame_id != prev_frame:
            print(f"Done frame {frame_id}")
            videoWriter.write(img)
            img = img_input.copy()
            prev_frame = frame_id
            prev_person = person_id - 1 #make sure construct key_points
        if person_id != prev_person:
            (f1_x, f1_y), (f2_x, f2_y) = get_points_of_foot(key_points)
            # print(f"frame {frame_id} person {prev_person} has ({f1_x}, {f1_y}), ({f2_x}, {f2_y})")
            fp = np.array([[f1_x, f1_y], [f2_x, f2_y]])
            fp = fp.reshape(-1, 1, 2).astype(np.float32)
            pointsOut = cv2.perspectiveTransform(fp, homography)
            dst1_x = int(pointsOut[0][0][0])
            dst1_y = int(pointsOut[0][0][1])
            dst2_x = int(pointsOut[1][0][0])
            dst2_y = int(pointsOut[1][0][1])
            # print(f"Draw ({dst1_y}, {dst1_x}) ({dst2_y}, {dst2_x})")
            img = cv2.line(img, (dst1_x, dst1_y), (dst2_x, dst2_y), (255, 0, 0), 2)
            key_points = [(x, y)]
            prev_person = person_id
        else:
            key_points.append((x, y))

videoWriter.release()
cv2.destroyAllWindows()
