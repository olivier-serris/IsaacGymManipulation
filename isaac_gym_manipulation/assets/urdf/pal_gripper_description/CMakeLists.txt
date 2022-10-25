cmake_minimum_required(VERSION 2.8.3)
project(pal_gripper_description)

find_package(catkin REQUIRED COMPONENTS
  xacro
)

###################################
## catkin specific configuration ##
###################################
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES pal_gripper_description
#  CATKIN_DEPENDS xacro
#  DEPENDS system_lib
)

foreach(dir gazebo meshes robots urdf)
    install(DIRECTORY ${dir}/
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
endforeach(dir)

#############
## Testing ##
#############

if(CATKIN_ENABLE_TESTING)
  find_package(rostest REQUIRED)
  add_rostest(test/test_gripper_test.test)
endif()

