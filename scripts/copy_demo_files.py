import os
import shutil

src_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results'))
dst_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'demo', 'models'))

os.makedirs(dst_dir, exist_ok=True)

for filename in os.listdir(src_dir):
    if filename.endswith('.joblib'):
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        if os.path.exists(dst_path):
            os.remove(dst_path)
        shutil.copy(src_path, dst_path)
        print(f"Copied: {filename}")

src_file = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'testing', 'unscaled_testing.csv'))
dst_dir2 = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'demo', 'data'))

os.makedirs(dst_dir2, exist_ok=True)

dst_file = os.path.abspath(os.path.join(dst_dir2, 'test_data.csv'))
if os.path.exists(dst_file):
    os.remove(dst_file)
shutil.copy(src_file, dst_file)
print(f"Copied: {src_file} to {dst_file}")