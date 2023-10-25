from setuptools import setup, find_packages

requirements = [
    'gym==0.21.0',
    'mujoco-py',
    'numpy>=1.18',
]

setup(
    name='simxarm',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    license=open('LICENSE').read(),
    zip_safe=False,
    description="Minimal xArm simulation environment for RL with a gym-style API.",
    author='Nicklas Hansen & Yanjie Ze',
    author_email='nihansen@ucsd.edu',
    url='https://github.com/nicklashansen/simxarm',
    keywords=['MuJoCo', 'Robotics', 'Reinforcement Learning', 'Gym'],
)
