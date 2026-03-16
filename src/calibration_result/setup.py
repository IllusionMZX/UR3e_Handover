import os
from glob import glob
from setuptools import setup

package_name = 'calibration_result'

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
