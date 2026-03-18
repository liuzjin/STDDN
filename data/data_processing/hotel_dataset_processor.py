from collections import defaultdict

import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm
import argparse

import os

def get_args():
    parser = argparse.ArgumentParser(description='New dataset interpolation processor')
    parser.add_argument('-i', '--input', type=str, default='./data_origin/hotel',
                        help='path to the input npy file, containing list of trajectories')

    parser.add_argument('-t', '--time', type=float, default='0',
                        help='begining of time snippet to save. Recommendation parameter:0,  1806, 3612, 5418')
    parser.add_argument('-d', '--duration', type=float, default='903',
                        help='length of time snippet to save')
    parser.add_argument('--dt', type=float, default=0.04,
                        help='interpolation time step in seconds')
    args = parser.parse_args()
    return args

def get_raw_data(data_name = 'hotel'):
    trajectories = defaultdict(list)
    for mode in ['train', 'val', 'test']:
        # 读取txt文件并转换为数据
        file_path = f'./data_origin/{data_name}/{mode}/biwi_{data_name}_{mode}.txt'  # 替换成你的文件路径
        with open(file_path, 'r') as file:
            for line in file:
                data = line.strip().split('\t')
                time = int(float(data[0]))  # 时间
                person_id = int(float(data[1]))  # 人的编号
                x, y = float(data[2]), float(data[3])  # 坐标

                # 将每个人的轨迹保存到字典
                if (x, y, time) not in trajectories[person_id]:
                    trajectories[person_id].append((x, y, time))
    trajectory_list = [traj for traj in trajectories.values()]
    return trajectory_list

def interpolate_trajectories(data, frame_range, dt=0.04):
    new_trajectories = []

    for traj in tqdm(data):
        traj = np.array(traj)  # shape: [N, 3]
        if traj.shape[0] < 2:
            continue  # 无法插值
        t_min, t_max = traj[:, 2].min(), traj[:, 2].max()
        sample_times = np.arange(t_min, t_max + 1, dt)

        # 初始化新的轨迹
        new_traj = np.zeros((len(sample_times), 3))
        new_traj[:, 2] = sample_times

        try:
            f_x = interp1d(traj[:, 2], traj[:, 0], kind='cubic')
            f_y = interp1d(traj[:, 2], traj[:, 1], kind='cubic')
            new_traj[:, 0] = f_x(sample_times)
            new_traj[:, 1] = f_y(sample_times)
        except ValueError:
            # 如果点数太少，改用线性插值
            new_traj[:, 0] = np.interp(sample_times, traj[:, 2], traj[:, 0])
            new_traj[:, 1] = np.interp(sample_times, traj[:, 2], traj[:, 1])

        traj_pola = [(x, y, int(f / time_unit / 25)) for x, y, f in new_traj if
                ((f >= frame_range[0]) and (f <= frame_range[1]))]
        if len(traj_pola) > 0:
            new_trajectories.append(traj_pola)

    return new_trajectories

if __name__ == '__main__':
    args = get_args()
    time_range = (int(args.time), int(args.time + args.duration))
    frame_range = [time_range[0] * 2.5, time_range[1] * 2.5]
    print(f"Loading raw trajectories from {args.input}")
    raw_trajectories = get_raw_data('hotel')
    max_t = 0
    min_t = 1000000
    for tra_ in raw_trajectories:
        if max_t < max([t for _, _, t in tra_]):
            max_t = max([t for _, _, t in tra_])
        if min_t > min([t for _, _, t in tra_]):
            min_t = min([t for _, _, t in tra_])
    print(f"raw time {min_t}-{max_t/2.5}s")
    time_unit = 1.0/12.5
    meta = {
        "time_unit": time_unit,
        "version": "v2.2",
        "begin_time": time_range[0],
        "source": "HOTEL dataset"
    }

    print("Interpolating trajectories...")
    interpolated_trajectories = interpolate_trajectories(raw_trajectories, frame_range, dt=time_unit * 25)
    destination = []
    for traj in interpolated_trajectories:
        destination.append([(traj[-1][0], traj[-1][1], traj[-1][2])])


    savename = args.input + f"/HOTEL_Dataset_time{time_range[0]}-{time_range[1]}_timeunit{time_unit:.2f}"
    print(f"Saving interpolated trajectories to {savename}")
    data = np.array((meta, interpolated_trajectories, destination, []), dtype=object)
    np.save(savename + ".npy", data)
    print("Done.")
