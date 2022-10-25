^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package pal_gripper_description
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1.0.5 (2021-06-11)
------------------

1.0.4 (2021-03-29)
------------------

1.0.3 (2020-04-30)
------------------

1.0.2 (2019-06-11)
------------------

1.0.1 (2019-03-26)
------------------
* Merge branch 'add-sides' into 'erbium-devel'
  Add option to specify side of gripper
  See merge request robots/pal_gripper!7
* Fix missing dependency
* Merge branch 'fix_xacro_warning' into 'erbium-devel'
  fix xacro warning
  See merge request robots/pal_gripper!5
* rm usuless xacro property
* Added Sanity check
* deprecate upload_gripper.launch
* fix xacro.py deprecation warning
* fix xacro warning
  deprecated: xacro tags should be prepended with 'xacro' xml namespace.
  Use the following script to fix incorrect usage:
  find . -iname "*.xacro" | xargs sed -i 's#<\([/]\?\)\(if\|unless\|include\|arg\|property\|macro\|insert_block\)#<\1xacro:\2#g'
* Contributors: Jeremie Deray, Jordi Pages, Victor Lopez, davidfernandez

1.0.0 (2018-07-30)
------------------
* Merge branch 'fix-simulation-warnings' into 'erbium-devel'
  Fix simulation warnings
  See merge request robots/pal_gripper!6
* prepend missing 'xacro' tag
* remove link color discrepancy
* Contributors: Jordi Pages, Victor Lopez

0.0.13 (2018-04-13)
-------------------
* Merge branch 'add-tool-link' into 'dubnium-devel'
  Add tool link
  See merge request robots/pal_gripper!4
* Add tool link
* Contributors: Hilario Tome, Victor Lopez

0.0.12 (2018-02-20)
-------------------

0.0.11 (2018-01-24)
-------------------

0.0.10 (2018-01-24)
-------------------
* move scripts and config files from tiago_robot
* Contributors: Jordi Pages

0.0.9 (2016-10-14)
------------------
* Fixed problem with gripper_grasping_frame in wrong position
* fix maintainer
* 0.0.8
* Update changelog
* use box for fingers' collision model
* 0.0.7
* Update changelogs
* 0.0.6
* Update cahngelog
* remove grasping hack macro and tune friction
* update meshes and inertia matrices
* 0.0.5
* Update changelog
* Change gripper limit to 0.045
* 0.0.4
* Update changelgo
* Fix safety joint limit
* 0.0.3
* Update changelogs
* Added safety controller values
* 0.0.2
* Updated the changelog
* Added install rules
* Contributors: Adria Roig, Hilario Tome, Jordi Pages, Sam Pfeiffer, Victor Lopez

0.0.1 (2016-06-01)
------------------
* Initial version
* Contributors: Sam Pfeiffer
