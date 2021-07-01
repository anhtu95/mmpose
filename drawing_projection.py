import numpy as np
import cv2
import os

COLOR = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
         (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
         (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
         (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195),
         (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128)]


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


def get_points_of_head(list_points):
    min_y = 100000
    head_x = -1
    head_y = -1
    for p in list_points:
        if p[1] < min_y:
            min_y = p[1]
            head_x = p[0]
            head_y = p[1]
    return (head_x, head_y)


pts_src = np.array([
    [466, 86], # [723, 142],
    [24, 256], # [460, 254],
    [1097, 393], # [1093, 324],
    [2130, 231], # [1696, 230],
    [1669, 75], # [1430, 136],
    [1078, 102], # [1082, 157]
])

h_pts_src = np.array([
    [483, 33], # [731, 107],
    [45, 145], # [467, 208],
    [1126, 247], # [1125, 208],
    [2146, 110], # [1699, 208],
    [1681, 15],# [1435, 107],
    [1094, 39]# [1093, 107]
])

pts_dst = np.array([
    [6, 5],
    [6, 355],
    [319, 355],
    [632, 355],
    [632, 5],
    [319, 5],
])

out_video_root = "./output"

img_input = cv2.imread("/home/anhtu/Project/trento/computer-vision/mmpose/resources/basket_field.jpg")
img_src = cv2.imread("/home/anhtu/Project/trento/computer-vision/output/short_4quarto_pt2_mmpose_cropped/vis_frame_0.png")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videoWriter = cv2.VideoWriter(os.path.join(out_video_root, "projection_foot_head_cropped.mp4"), fourcc, 5.0,
                              (img_input.shape[1], img_input.shape[0]))

homography, status = cv2.findHomography(pts_src, pts_dst)
h_homography, _ = cv2.findHomography(h_pts_src, pts_dst)
font = cv2.FONT_HERSHEY_SIMPLEX


with open("./output/keypoints_cropped.txt", 'r') as f:
    prev_frame = 0
    prev_person = 0
    key_points = []
    img = img_input.copy()
    img_copy = img_input.copy()
    for line in f:
        info = line.split(" ")
        frame_id, person_id, x, y = int(info[0]), int(info[1]), int(info[2]), int(info[3])
        if person_id != prev_person or frame_id != prev_frame:
            (f1_x, f1_y), (f2_x, f2_y) = get_points_of_foot(key_points)
            img_src = cv2.putText(img_src, "f_"+str(person_id), (int((f1_x+f2_x)/2), int((f1_y+f2_y)/2)), font, 1, COLOR[person_id], 2)

            fp = np.array([[f1_x, f1_y], [f2_x, f2_y]])
            fp = fp.reshape(-1, 1, 2).astype(np.float32)
            pointsOut = cv2.perspectiveTransform(fp, homography)
            dst1_x = int(pointsOut[0][0][0])
            dst1_y = int(pointsOut[0][0][1])
            dst2_x = int(pointsOut[1][0][0])
            dst2_y = int(pointsOut[1][0][1])

            h_x, h_y = get_points_of_head(key_points)
            img_src = cv2.putText(img_src, "h_"+str(person_id), (h_x, h_y), font, 1, COLOR[person_id], 2)
            hp = np.array([[h_x, h_y]])
            hp = hp.reshape(-1, 1, 2).astype(np.float32)
            hPointsOut = cv2.perspectiveTransform(hp, h_homography)
            # hPointsOut = cv2.perspectiveTransform(hp, homography)
            h_dst1_x = int(hPointsOut[0][0][0])
            h_dst1_y = int(hPointsOut[0][0][1])
            # print(f"Draw ({dst1_y}, {dst1_x}) ({dst2_y}, {dst2_x})")
            img = cv2.putText(img, "f_"+str(person_id), (int((dst1_x+dst1_x)/2), int((dst1_y+dst2_y)/2)), font, 0.5, COLOR[person_id], 1)
            img = cv2.putText(img, "h_"+str(person_id), (h_dst1_x, h_dst1_y), font, 0.5, COLOR[person_id], 1)
            img_copy = cv2.putText(img_copy, "f_" + str(person_id), (int((dst1_x + dst1_x) / 2), int((dst1_y + dst2_y) / 2)),
                              font, 0.5, COLOR[person_id], 1)
            img_copy = cv2.putText(img_copy, "h_" + str(person_id), (h_dst1_x, h_dst1_y), font, 0.5, COLOR[person_id], 1)

            img_copy = cv2.line(img_copy, (dst1_x, dst1_y), (dst2_x, dst2_y), COLOR[person_id], 2)
            img_copy = cv2.circle(img_copy, (h_dst1_x, h_dst1_y), 2, COLOR[person_id], 2)
            img = cv2.line(img, (dst1_x, dst1_y), (dst2_x, dst2_y), COLOR[person_id], 2)
            img = cv2.circle(img, (h_dst1_x, h_dst1_y), 2, COLOR[person_id], 2)

            key_points = [(x, y)]
            prev_person = person_id
        else:
            key_points.append((x, y))
        if frame_id != prev_frame:
            print(f"Done frame {frame_id}")
            videoWriter.write(img)
            cv2.imwrite(f"./output/basket_field/frame_{prev_frame}.png", img_copy)
            cv2.imwrite(f"./output/key_points/frame_{prev_frame}.png", img_src)
            img = img_input.copy()
            img_copy = img_input.copy()
            img_src = cv2.imread(f"/home/anhtu/Project/trento/computer-vision/output/short_4quarto_pt2_mmpose_cropped/vis_frame_{frame_id}.png")
            prev_frame = frame_id
            # prev_person = person_id - 1  # make sure construct key_points

videoWriter.release()
cv2.destroyAllWindows()
