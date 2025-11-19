import os

instance_base_directory = '/data2/users/lanjie/proj/SSL_workplace/dataset/'

imgs_directory1 = os.path.join(instance_base_directory, 'Vaihingen/val/ori_image/')
masks_directory1 = os.path.join(instance_base_directory, 'Vaihingen/val/ori_label/')

output_file = '/data2/users/lanjie/proj/SSL_workplace/contexmatch/splits/vaihingen/val.txt'

# 支持的图像文件扩展名
image_extensions = {'.jpg', '.jpeg', '.png', '.tif'}


def get_image_mask_pairs(imgs_directory, masks_directory):
    pairs = []
    img_files = sorted(os.listdir(imgs_directory1))
    mask_files = sorted(os.listdir(masks_directory))

    print('img_files', img_files[:5])
    print('mask_files', mask_files[:5])
    
    # 假设img和mask文件一一对应，只是名称不同，这里通过索引关联
    for img_filename, mask_filename in zip(img_files, mask_files):
        rel_img_path = os.path.relpath(os.path.join(imgs_directory, img_filename), instance_base_directory)
        rel_mask_path = os.path.relpath(os.path.join(masks_directory, mask_filename), instance_base_directory)
        
        if os.path.exists(os.path.join(instance_base_directory, rel_mask_path)):
            pairs.append(f"{rel_img_path} {rel_mask_path}")
    
    return pairs

# 获取图像和分割标签的路径对
pairs = get_image_mask_pairs(imgs_directory1, masks_directory1)

# 输出调试信息
print(f"Total pairs collected: {len(pairs)}")

# 读取现有文件内容（如果文件存在）
existing_pairs = set()
if os.path.exists(output_file):
    with open(output_file, 'r') as f:
        existing_pairs = set(f.read().splitlines())

# 输出调试信息
print(f"Existing pairs in file: {len(existing_pairs)}")

# 合并现有的对和新的对，并去重
all_pairs = sorted(set(existing_pairs).union(pairs))

# 输出调试信息
print(f"Total unique pairs to write: {len(all_pairs)}")

# 创建目录，如果不存在
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# 将所有对写入输出文件
with open(output_file, 'w') as f:
    for pair in all_pairs:
        f.write(f"{pair}\n")

print("Finished writing pairs to file.")