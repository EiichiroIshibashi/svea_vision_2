from glob import glob
from setuptools import find_packages, setup

package_name = "svea_vision_experimental"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
    ],
    install_requires=[
        "setuptools<70.0.0",
        "ultralytics==8.2.0",
        "numpy<2.0",
    ],
    zip_safe=True,
    maintainer="Local Experiment",
    maintainer_email="local-experiment@example.com",
    description="Experimental dual-model object detection for SVEA vision.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "dual_object_detect = svea_vision_experimental.nodes.object.dual_object_detect:main",
        ],
    },
)
