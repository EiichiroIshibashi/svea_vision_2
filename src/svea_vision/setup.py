from setuptools import find_packages, setup
from glob import glob

package_name = 'svea_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=[
        'setuptools<70.0.0',
        'ultralytics==8.2.0',
        'filterpy',
        'numba',
        'pandas',
        'numpy<2.0'
    ],
    zip_safe=True,
    maintainer='Yunhao Xu',
    maintainer_email='yunhaox@kth.se',
    description='The svea vision package',
    license='MIT',
    # tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'aruco_detect = svea_vision.nodes.aruco.aruco_detect:main',
            'static_image_publisher = svea_vision.nodes.image.static_image_publisher:main',
            'object_detect = svea_vision.nodes.object.object_detect:main',
            'object_pose = svea_vision.nodes.object.object_pose:main',
            # 'person_state_estimation = svea_vision.nodes.person.person_state_estimation:main',
            'person_state_estimation = svea_vision.nodes.person.person_state_estimation_xyvv:main',
            'sidewalk_mapper = svea_vision.nodes.sidewalk.sidewalk_mapper:main',
            'detection_splitter = svea_vision.nodes.object.detection_splitter:main',
            'pedestrian_flow_estimate = svea_vision.nodes.person.pedestrian_flow_estimate:main',
            'segment_anything = svea_vision.nodes.sidewalk.segment_anything:main',
        ],
    },
)
