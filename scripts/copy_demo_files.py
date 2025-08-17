import os
import shutil

src_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), 'results'))
dst_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), 'demo', 'models'))

os.makedirs(dst_dir, exist_ok=True)

for filename in os.listdir(src_dir):
    if filename.endswith('.joblib'):
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        shutil.copy(src_path, dst_path)
        print(f"Copied: {filename}")

src_file = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), 'data', 'testing', 'card_transdata_part2.csv'))
dst_file = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), 'demo', 'data', 'test_data.csv'))

shutil.copy(src_file, dst_file)
print(f"Copied: {src_file} to {dst_file}")