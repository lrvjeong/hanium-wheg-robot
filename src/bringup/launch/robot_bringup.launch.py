import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    # 라이다 드라이버
    cyglidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('cyglidar_d2_ros2'),
                'launch',
                'cyglidar.launch.py'
            )
        )
    )

    # 센서 퓨전 노드
    sensor_fusion_node = Node(
        package='perception_pkg',
        executable='sensor_fusion_node',
        name='sensor_fusion_node',
        output='screen'
    )

    # ToF(근거리) 노드
    tof_node = Node(
        package='perception_pkg',
        executable='tof_node',
        name='tof_node',
        output='screen'
    )

    # FSM 노드
    mode_fsm_node = Node(
        package='control_pkg',
        executable='mode_fsm_node',
        name='mode_fsm_node',
        output='screen'
    )

    # 모터 인터페이스 노드
    motor_interface_node = Node(
        package='motor_pkg',
        executable='motor_interface_node',
        name='motor_interface_node',
        output='screen'
    )

    # 모터/안전 통합 노드 (다이나믹셀 + 아두이노 서보/IMU 전담, 실제 하드웨어 제어)
    safety_stop_node = Node(
        package='motor_pkg',
        executable='safety_stop_node',
        name='safety_stop_node',
        output='screen'
    )

    return LaunchDescription([
        cyglidar_launch,
        sensor_fusion_node,
        tof_node,
        mode_fsm_node,
        motor_interface_node,
        safety_stop_node,        # 수정
    ])
