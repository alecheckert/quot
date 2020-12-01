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
	entry_points = {
		'console_scripts': [
			'quot=quot.gui.__main__:cli',
            'quot-track=quot.__main__:batch_track'
		],
	},
)
