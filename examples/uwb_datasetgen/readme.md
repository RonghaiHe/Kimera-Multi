# uwb 数据生成

在gt中依次选取一个为源节点，其他为目标节点，根据源节点首尾时间重新生成时间戳，计算若干个源节点到目标节点的距离，生成uwb数据。
每一个bot_id和uwb的全局id为定值，其中bot_id由botids中名称的index决定，uwb的全局id由安装的bot_id和uwb的本地id决定。
生成msg后根据时间戳，以输入的topic格式插入指定的bag文件中，
使用时修改以下关键后执行即可：
注意：插入后若需要删除某topic可参考使用[rosbag-topic-remove](https://github.com/IamPhytan/rosbag-topic-remove/tree/main)
mode 一定要设置为'a',否则会覆盖原有数据

主程序[bagprocess_bag.py](bagprocess_bag.py)的参数说明：
- `gt_path_list`: 真实数据路径列表，其中每一个文件应当满足命名 'gt_path_list/{bot_id}_gt_odom.csv',如果gt文件格式不同，修改58-59行解析代码
- `topic_format_in`: 话题格式，默认是"/{src_name}/{local_uwb_id}/uwb_distance", 其中src_name是源节点名称，local_uwb_id是本地uwb id，格式化参数不可缺少与修改，命名格式可修改
- `bag_format_in`: 生成数据目标写入的bag文件格式，默认是"campus_tunnels_12_07/12_07_{src_name}.bag"，其中src_name是源节点名称，格式化参数不可缺少与修改，命名格式可修改
- `uwb_position`: 每一个bot的uwb相对于bot的坐标系的相对位置，为(n,3)的数组
- `publish_feq`: 以此频率重新生成时间与距离数据
- `noise_err`: 噪声标准差
  
