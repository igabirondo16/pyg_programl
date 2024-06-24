from setuptools import setup, find_packages

# Function to read requirements from requirements.txt
def read_requirements():
    with open('./requirements.txt') as req_file:
        return req_file.read().splitlines()

setup(
    name='load_balancer_programl',
    version='0.1.0',
    packages=find_packages(),
    install_requires=read_requirements(),
    author='Iñigo Gabirondo López',
    author_email='igabirondo13@gmail.com',
    description='Deep Learning Load Balancer based on ProGraML for a CPU+GPU system.',
    long_description=open('./README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/igabirondo16/pyg_programl',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
