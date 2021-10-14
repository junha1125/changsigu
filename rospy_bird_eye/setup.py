#!/usr/bin/env python3

from distutils.core import setup
from catkin_pkg.package import parse_package_for_distutils

package_info = parse_package_for_distutils()
package_info['packages'] = ['bird_eye']
package_info['package_dir'] = {'bird_eye': '/home/jetson/catkin_ws/src/rospy_bird_eye/bird_eye'}
package_info['install_requires'] = []

setup(**package_info)
