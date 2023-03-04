"""
setup.py
"""
import setuptools

setuptools.setup(
	name="quot",
	version="3.0",
	packages=setuptools.find_packages(),
	author="Alec Heckert",
	author_email="aheckert@berkeley.edu",
	description="GUI for single molecule tracking",
    install_requires=[
        "dask",
        "matplotlib",
        "munkres",
        "nd2reader==3.2.1",
        "nose2",
        "numpy",
        "pandas",
        "scikit-image",
        "scipy",
        "seaborn",
        "tifffile",
        "toml",
        "tqdm",
        "pyqtgraph",
        "pyside6",
    ],
	entry_points = {
		'console_scripts': [
			'quot=quot.gui.__main__:cli',
            'quot-track=quot.__main__:batch_track',
            'quot-config=quot.__main__:make_naive_config'
		],
	},
)
