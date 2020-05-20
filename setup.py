"""
setup.py
"""
import setuptools

setuptools.setup(
	name="quot",
	version="2.0",
	packages=setuptools.find_packages(),
	author="Alec Heckert",
	author_email="aheckert@berkeley.edu",
	description="GUI for single molecule tracking",
	entry_points = {
		'console_scripts': [
			'quot=quot.__main__:cli',
		],
	},
)
