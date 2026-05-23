import os
from glob import glob
from pathlib import Path

from setuptools import setup

package_name = 'calibration_result'
package_root = Path(__file__).resolve().parent
resource_dir = package_root / 'resource'
resource_marker = resource_dir / package_name

# ament_python packages expect a resource index marker file.
# Create it on demand so colcon can install this package even if the marker
# was not committed with the package skeleton.
resource_dir.mkdir(exist_ok=True)
resource_marker.touch(exist_ok=True)

setup(
    name=package_name,
    version='0.0.0',
    packages=[],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'ur3e_eye_in_hand'), glob('ur3e_eye_in_hand/*.launch.py')),
        (os.path.join('share', package_name, 'ur3e_eye_to_hand'), glob('ur3e_eye_to_hand/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Miao Zixiang',
    maintainer_email='miao.zixiang@foxmail.com',
    description='Calibration results and launch files',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [],
    },
)
