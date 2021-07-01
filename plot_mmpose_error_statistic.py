import matplotlib.pyplot as plt

missing_player = []
avg_missing_point = []

with open("./output/keypoints_cropped.txt", 'r') as f:
    prev_frame = 0
    prev_person = 0
    nb_key_point = 0
    nb_missing_key_point = 0
    nb_person_miss_point = 0
    nb_person = 0
    for line in f:
        info = line.split(" ")
        frame_id, person_id, x, y = int(info[0]), int(info[1]), int(info[2]), int(info[3])
        if person_id != prev_person or frame_id != prev_frame:
            if nb_key_point != 17:
                nb_missing_key_point += (17-nb_key_point)
                nb_person_miss_point += 1
            nb_key_point = 1
            prev_person = person_id
            nb_person += 1
        else:
            nb_key_point += 1

        if frame_id != prev_frame:
            print(f"DONE frame {frame_id-1}")
            prev_frame = frame_id
            missing_player.append(11-nb_person)
            if nb_person_miss_point != 0:
                avg_missing_point.append(nb_missing_key_point/nb_person_miss_point)
            else:
                avg_missing_point.append(0)
            nb_person = 0
            nb_person_miss_point = 0
            nb_missing_key_point = 0

plt.plot(range(len(missing_player)), missing_player, 'o-r')
plt.xlabel("Frame")
plt.title("Number of missing player per frame")
plt.show()

print(avg_missing_point)
plt.plot(range(len(avg_missing_point)), avg_missing_point, 'o-g')
plt.xlabel("Frame")
plt.title("Average number of missing point per frame")
plt.show()