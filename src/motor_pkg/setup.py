from setuptools import find_packages, setup

package_name = 'motor_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tpwjd',
    maintainer_email='sjkim26999@gmail.com',
    description='Motor interface node',
    license='TODO: License declaration',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'motor_interface_node = motor_pkg.motor_interface_node:main',
        ],
    },
)
