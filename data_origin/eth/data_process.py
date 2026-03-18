from collections import defaultdict
import numpy as np

data_name = 'eth'
for mode in ['train', 'val', 'test']:
    # 读取txt文件并转换为数据
    file_path = f'./data_origin/{data_name}/{mode}/biwi_{data_name}_{mode}.txt'  # 替换成你的文件路径
    trajectories = defaultdict(list)

    with open(file_path, 'r') as file:
        for line in file:
            data = line.strip().split('\t')
            time = int(float(data[0]))  # 时间
            person_id = int(float(data[1]))  # 人的编号
            x, y = float(data[2]), float(data[3])  # 坐标

            # 将每个人的轨迹保存到字典
            trajectories[person_id].append((x, y, time))

    # 将字典转换为列表形式
    meta_data = {'dataset': 'eth', 'version': 'v2.2', 'time_unit': 0.4}
    trajectory_list = [traj for traj in trajectories.values()]
    destinations = [[traj[-1]] for traj in trajectories.values()]
    obstacles = []  # 假设没有障碍物数据

    # 将数据打包成一个列表
    # data_to_save = (meta_data, trajectory_list, destinations, obstacles)
    data_to_save = np.array([meta_data, trajectory_list, destinations, obstacles], dtype=object)
    # 保存为npy文件，确保允许pickle类型
    np.save(f'./data_origin/{data_name}/{data_name}_{mode}.npy', data_to_save, allow_pickle=True)
