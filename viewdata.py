import zarr
import numpy as np
import visualizer
import cv2

dataset_path = '/opt/projects/3D-Diffusion-Policy/data/realdex_pour.zarr'
# dataset_path = '/opt/projects/3D-Diffusion-Policy/data/aubo.zarr'
dataset_path = '/opt/projects/3D-Diffusion-Policy/data/realdex_drill.zarr'
dataset_path = '/opt/projects/diffusion_policy/data/hang316.zarr'
# dataset_path = '/opt/projects/Improved-3D-Diffusion-Policy/training_data_example'

zarr_file = zarr.open(dataset_path, mode='r')
zarr_meta_info = zarr_file['meta']
episode_ends = zarr_meta_info['episode_ends']
print(f"total episode {len(episode_ends)} \n, episodes: {list(episode_ends)}")
# data file
zarr_file = zarr_file['data']
point_cloud = zarr_file['point_cloud'][208]
# print(zarr_file['point_cloud'][0][0:50])
# visualizer.visualize_pointcloud(point_cloud)

img = zarr_file['image'][0]
cv2.imwrite('img.jpg', img)

# img_len = len(zarr_file['img'])
# for i in range(img_len):
#     img = zarr_file['img'][i]
#     if i == 0:
#         height, width = img.shape[:2]
#         video_writer = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
#     video_writer.write(img)
# video_writer.release()

depth = zarr_file['depth'][0]
depth = depth[:-1,1:]
max = depth.max()
min = depth.min()
depth = ((depth - min) / (max - min) * 255).astype(np.uint8)
cv2.imwrite('depth.jpg', depth)

actions = zarr_file['action']
states = zarr_file['state']
print(actions.shape)
print(states.shape)
for i in range(50):
    print(f"states: {states[i]}")
    print(f"actions: {actions[i]}")

for key in zarr_file.keys():
        print(f"Array name: {key}")
        array = zarr_file[key]
        print(f"Array shape: {array.shape}")
        print(f"Array dtype: {array.dtype}")
        print(f'min {np.min(array)} max {np.max(array)}')
        if key =='action':
            print(f'gripper in action min {np.min(array[:,-1])} max {np.max(array[:,-1])}, unique {np.unique(array[:,-1])}')
            for i in range(6):
                print(f'joint{i+1} in action min {np.min(array[:, i])} max {np.max(array[:,i])}, unique {np.unique(array[:,i])}')
        if key =='state':
            print(f'gripper in state min {np.min(array[:,-1])} max {np.max(array[:,-1])}, unique {np.unique(array[:,-1])}')
            for i in range(6):
                print(f'joint{i+1} in state min {np.min(array[:, i])} max {np.max(array[:,i])}, unique {np.unique(array[:,i])}')
        # partial_data = array[:1]
        # print(f"partial data: {partial_data}")