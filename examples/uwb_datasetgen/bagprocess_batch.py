

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import bisect
from pyLieAlg import so3, SO3, SE3
import tqdm
from nlink_parser.msg import LinktrackNodeframe2, LinktrackNode2

# ------------------ 输入参数 ---------------
# 计算src 的bot到rec的相对距离序列
# src_path = "1207/acl_jackal_gt_odom.csv"
# rec_path_list = [
#     "1207/acl_jackal2_gt_odom.csv",
#     "1207/sparkal1_gt_odom.csv",
#     "1207/sparkal2_gt_odom.csv",

# ]

gt_path_prefix = "/media/pc/Data/1_mproject/Kimera/Dataset/1207"

gt_path_list = [
    gt_path_prefix + "/acl_jackal_gt_odom.csv",
    gt_path_prefix + "/acl_jackal2_gt_odom.csv",
    gt_path_prefix + "/hathor_gt_odom.csv"
]

topic_format_in = "/{src_name}/uwb_distance/{local_uwb_id}"

# bag_path = "12_07_acl_jackal.bag"

bag_format_in = "/media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/campus_outdoor_10_14/10_14_{src_name}.bag"

# from left_camera to body: [-0.0302200000733137, 0.00740000000223517, 0.0160199999809265]
t_b_left_camera = [-0.0302200000733137,
                   0.00740000000223517, 0.0160199999809265]
uwb_position_wrt_left_camera = [
    [-0.04, 0.035, 0.005],
    [0.022, -0.019, -0.051],
    [0.141, 0.04, -0.068]
]

uwb_position = [[a + b for a, b in zip(t_b_left_camera, pos)]
                for pos in uwb_position_wrt_left_camera]

publish_feq = 50.0  # Hz
noise_effect = 0.01  # 高斯噪声的标准差占原始值的比例
noise_err = 0.0382
mode = "a"
test_output = True

# 传感器误差率
senser_error_rate = 0.01
# 传感器误差生成器
def senser_generator():
    # 生成 N(2,0.1) 的误差
    return np.random.normal(2, 0.1)



# bot间多径效应误差
nloss_error_inter_bot = 0.005
# 单uwb间多径效应误差率
nloss_error_inter_uwb = 0.005

# 多径效应误差生成器
def nloss_generator(rd):
    # 生成 rd 的n 倍误差，其中n 为0-1均匀分布
    n = np.random.uniform(0, 1)
    # 生成 rd 的n 倍误差
    return rd * n

botids = [
    "acl_jackal",
    "acl_jackal2",
    "sparkal1",
    "sparkal2",
    "hathor",
    "thoth",
]

# ------------------ 自动计算参数 ---------------

def process_once(src_path, rec_path_list, topic_fomat, bag_path, uwb_position, publish_feq, noise_effect):
    # ------------------ 自动计算参数 ---------------
    # 以下部分无需修改
    rec_path = "1207/sparkal1_gt_odom.csv"
    # botids = [
    #     # "acl_jackal",
    #     # "acl_jackal2",
    #     # "sparkal1",
    #     # "sparkal2",
    #     # "hathor",
    #     # "thoth",
    # ]




    uwb_num = len(uwb_position)


    rec_path = rec_path_list[0] if len(rec_path_list) > 0 else rec_path
    src_name = src_path.split("/")[-1].split("_gt_odom")[0]
    rec_name = rec_path.split("/")[-1].split("_gt_odom")[0]

    rec_id = botids.index(rec_name)
    src_id = botids.index(src_name)

    print("src_name:", src_name)
    print("rec_name, rec_id:", rec_name, rec_id)
    topic = "uwb_distance"
    full_topic = "/"+src_name+"/"+rec_name+"/"+topic
    # print("full_topic:", full_topic)

    # import rosbag
    # with rosbag.Bag(bag_path, 'r') as bag:
    #     # 查看bag信息
    #     print("bag info:")
    #     print(bag.get_type_and_topic_info())

    # exit()

    # ------------------- 脚本开始 -------------------

    class LinkNode:
        '''
        uint8 role
        uint8 id    # 测量ID
        float32 dis
        float32 fp_rssi # 测量强度 初始化rx_rssi=-82，fp_rssi=-76
        float32 rx_rssi
        '''

        def __init__(self, id=-1, dis=-1, fp_rssi=-76, rx_rssi=-82):
            self.role = -1
            self.id = id
            self.dis = dis
            self.fp_rssi = fp_rssi
            self.rx_rssi = rx_rssi

    class LinktrackNodeframe:
        '''
            uint8 role              
            uint8 id                # 本机ID
            uint32 local_time       # unused
            uint32 system_time      # unused
            float32 voltage         # unused
            float32[3] pos_3d       # unused
            float32[3] eop_3d       # unused
            float32[3] vel_3d       # unused
            float32[3] angle_3d     # unused
            float32[4] quaternion   # unused
            float32[3] imu_gyro_3d  # unused
            float32[3] imu_acc_3d   # unused
            LinktrackNode2[] nodes  # 即上面的LinkNode ,其中ID同时标识了在列表中的index
        '''

        def __init__(self, id=-1):
            self.role = -1
            self.id = id
            self.local_time = 0
            self.system_time = 0
            self.voltage = 0.0
            self.pos_3d = [0.0, 0.0, 0.0]
            self.eop_3d = [0.0, 0.0, 0.0]
            self.vel_3d = [0.0, 0.0, 0.0]
            self.angle_3d = [0.0, 0.0, 0.0]
            self.quaternion = [1.0, 0.0, 0.0, 0.0]
            self.imu_gyro_3d = [0.0, 0.0, 0.0]
            self.imu_acc_3d = [9.8, 9.8, 9.8]
            self.nodes = []

    src_gt = pd.read_csv(src_path)
    rec_gt = pd.read_csv(rec_path)

    def resample(resample_time_list, rtime_list, rx, ry, rz,
                 rqw, rqx, rqy, rqz,
                 ):
        '''
            根据start和end时间戳与rx,ry,rz的长度，生成新的x,y,z列表,
            再根据resample_time_list重新采样,若resample_time_list之外，则填充-1

            resample_time_list: 重新采样的时间戳列表
            rtime_list: 原始时间戳列表
            rx, ry, rz: 原始坐标列表
            rqw, rqx, rqy, rqz: 原始四元数列表
        '''
        interp_x = interp1d(rtime_list, rx, kind='linear',
                            fill_value=-1, bounds_error=False)
        interp_y = interp1d(rtime_list, ry, kind='linear',
                            fill_value=-1, bounds_error=False)
        interp_z = interp1d(rtime_list, rz, kind='linear',
                            fill_value=-1, bounds_error=False)

        # resample_data = {
        #     "x": interp_x(resample_time_list),
        #     "y": interp_y(resample_time_list),
        #     "z": interp_z(resample_time_list),
        # }
        # resample_data = np.array([interp_x(resample_time_list), interp_y(resample_time_list), interp_z(resample_time_list)])
        # resample_data: x, y, z, qw, qx, qy, qz
        # 每一个resample_time_list的值，找到rtime_list中最大的小于等于它的值
        indices = []
        for time in resample_time_list:
            index = bisect.bisect_right(rtime_list, time) - 1
        # 如果time在rtime_list的范围内，则index为0到len(rtime_list)-2 , 即至少要有一个比当前值大
            if index < 0 or index >= len(rtime_list) - 1:
                indices.append(-1)
            else:
                indices.append(index)

        # 对位姿重插值
        # R_tr = R_tc * exp( dt_src / dt * log(R_tc^-1 * R_tc+dt) )
        qw = np.full(len(resample_time_list), fill_value=-1.0)
        qx = np.full(len(resample_time_list), fill_value=-1.0)
        qy = np.full(len(resample_time_list), fill_value=-1.0)
        qz = np.full(len(resample_time_list), fill_value=-1.0)
        for i, time in enumerate(resample_time_list):
            index = indices[i]
            if index < 0 or index >= len(rtime_list):
                continue
            # 计算当前时间戳对应的四元数
            i_r = SO3(
                quaternion=[rqw[index], rqx[index], rqy[index], rqz[index]],
            )
            i_r_dt = SO3(
                quaternion=[rqw[index + 1], rqx[index + 1],
                            rqy[index + 1], rqz[index + 1]],
            )
            dt = rtime_list[index + 1] - rtime_list[index]
            dt_src = time - rtime_list[index]

            logRR = i_r.inverse() * i_r_dt
            logRR = logRR.log().numbermultiply(dt_src / dt)
            expRR = logRR.exp()
            i_tr = i_r * expRR
            i_tr_q = i_tr.quaternion()
            qw[i] = i_tr_q[0]
            qx[i] = i_tr_q[1]
            qy[i] = i_tr_q[2]
            qz[i] = i_tr_q[3]

        resample_data = np.array([
            interp_x(resample_time_list),
            interp_y(resample_time_list),
            interp_z(resample_time_list),
            qw, qx, qy, qz
        ])
        return resample_data

    def uwb_position_transform(us_position, bot_trans, bot_rotation):
        '''
            us_position: bot坐标系下若干个uwb坐标(默认n=3个uwb,每个uwb 3个坐标 dx,dy,dz)
            bot_trans: bot在世界坐标系下的坐标 (3,)
            bot_rotation: bot在世界坐标系下的旋转矩阵 (3,3)
            @return: 3个uwb坐标在世界坐标系下的坐标  
        '''

        us_globle_position = np.zeros((uwb_num, 3))
        # 检查bot_trans是否全为-1，如果是，则返回[-1, -1, -1]
        if np.all(bot_trans == -1):
            us_globle_position.fill(-1)
            return us_globle_position


        for i in range(uwb_num):
            rp = us_position[i]
            # 计算uwb在世界坐标系下的坐标
            gp = bot_trans + bot_rotation @ rp
            # print("bot_trans", bot_trans)
            # print("bot_rotation", bot_rotation)
            # print("rp", rp)
            # print("gp", gp)
            us_globle_position[i] = gp
        return us_globle_position

    def rd_calculate_withnoise(src_us_global, rec_us_global,
                               if_nloss, if_senser_error,
                               ):
        '''
            计算相对距离
            src_us_global: src在世界坐标系下的坐标 (3,)
            rec_us_global: rec在世界坐标系下的坐标 (3,)
        '''
        # 检查src_us_global和rec_us_global中是否存在-1 u如果有，则返回-1
        if np.any(src_us_global == -1) or np.any(rec_us_global == -1):
            return -1.0

        # 计算相对距离
        rd = np.linalg.norm(src_us_global - rec_us_global)

        # 添加标准差固定的高斯噪声
        rdr = rd + np.random.normal(0, noise_err)

        # 添加传感器误差
        if if_senser_error:
            # 传感器误差
            rdr = rd + senser_generator()
        # 添加多径效应误差
        if if_nloss:
            # 多径效应误差
            rdr = rdr + nloss_generator(rd)

        return rdr


    '''
        根据输入publish_feq，重新采样src时间戳与坐标,并生成相对距离
    '''
    import rosbag
    import rospy
    # 根据publish_feq重新采样src时间戳
    publish_time_list = np.arange(
        src_gt.iloc[0]["#timestamp_kf"],
        src_gt.iloc[-1]["#timestamp_kf"],
        1.0 / publish_feq * 1e9
    )

    src_resample = resample(
        publish_time_list,
        np.array(src_gt["#timestamp_kf"]).astype(np.float64),
        np.array(src_gt["x"]).astype(np.float64),
        np.array(src_gt["y"]).astype(np.float64),
        np.array(src_gt["z"]).astype(np.float64),
        np.array(src_gt["qw"]).astype(np.float64),
        np.array(src_gt["qx"]).astype(np.float64),
        np.array(src_gt["qy"]).astype(np.float64),
        np.array(src_gt["qz"]).astype(np.float64)
    )

    # df = pd.DataFrame(src_resample.T, columns=["x", "y", "z", "qw", "qx", "qy", "qz"])
    # df.to_csv(f"src_resample.csv", index=False)
    # exit()

    botnum = len(botids)
    # botnum 个 "x", "y", "z", "qw", "qx", "qy", "qz" 时间序列，即 botnum * 7 * len(publish_time_list)
    resample_rec_data = np.full((botnum, 7, len(publish_time_list)), -1.0)
    print("resample_rec_data.shape:", resample_rec_data.shape)

    # 遍历rec_path_list,读取对应gt，并根据publish_feq重新采样
    for rec_path in rec_path_list:
        rec_gt = pd.read_csv(rec_path)
        rname = rec_path.split("/")[-1].split("_gt_odom")[0]
        rid = botids.index(rname)
        print("rec_name, rec_id:", rname, rid)
        # 根据publish_feq重新采样rec时间戳,  超出部分x y z为-1
        rec_resample = resample(
            publish_time_list,
            np.array(rec_gt["#timestamp_kf"]).astype(np.float64),
            np.array(rec_gt["x"]).astype(np.float64),
            np.array(rec_gt["y"]).astype(np.float64),
            np.array(rec_gt["z"]).astype(np.float64),
            np.array(rec_gt["qw"]).astype(np.float64),
            np.array(rec_gt["qx"]).astype(np.float64),
            np.array(rec_gt["qy"]).astype(np.float64),
            np.array(rec_gt["qz"]).astype(np.float64)
        )
        # # 将rec_resample中的数据写入文件检查
        # df = pd.DataFrame(rec_resample.T, columns=["x", "y", "z"])
        # df.to_csv(f"rec_resample_{rname}.csv", index=False)
        # exit()

        # 将rec_resample中的数据写入resample_rec_data
        resample_rec_data[rid] = rec_resample

    # 依时间计算相对距离
    # from msg import LinktrackNode2



    NodeFrameTimeList = []
    LinktrackNodeframeTimeList = []
    for i in tqdm.tqdm(range(len(publish_time_list))):
        time = publish_time_list[i]

        NodeFrameList = []
        LinktrackNodeframeList = []

        # uwb是否出现传感器误差
        if_uwberror_list = []

        for src_uid in range(uwb_num):
            # 本机存储uwb_num 个 NodeList ,index对应为本机uwb的本地序号，
            # frame2内对应的全局id 为 src_id * uwb_num + src_uid
            NodeList = []
            # 每一个src_uid构造一个LinktrackNodeframe2 msg，其id由当前全局uid决定
            timestamp = rospy.Time.from_sec(time * 1e-9)
            NodeFrame = LinktrackNodeframe2(
                id=src_uid + src_id * uwb_num, stamp=timestamp)

            for boti in range(len(botids)):
                for num in range(uwb_num):
                    # id = boti * uwb_num + num
                    NodeList.append(LinkNode(id=boti, dis=-1.0,
                                                fp_rssi=-76, rx_rssi=-82))
                    NodeFrame.nodes.append(LinktrackNode2(
                        role=0,
                        id=boti * uwb_num + num,
                        dis=-1.0,
                        fp_rssi=-76,
                        rx_rssi=-82
                    ))
            if test_output:
                NodeFrameList.append(NodeList)
            LinktrackNodeframeList.append(NodeFrame)
            if_uwberror = np.random.uniform(0, 1) < senser_error_rate
            if_uwberror_list.append(if_uwberror)

        src_us_global = uwb_position_transform(
            np.array(object=uwb_position),
            np.array(src_resample[0:3, i]),  # botid 0,1,2
            SO3(
                quaternion=[src_resample[3, i], src_resample[4, i], src_resample[5, i], src_resample[6, i]]
            ).matrix()
        )
        # 遍历所有的bot
        for botid in range(botnum):
            # 结算是否出现bot间的多径效应
            if_inter_bot = np.random.uniform(0, 1) < nloss_error_inter_bot

            if botid == src_id:
                # 自距离数据生成
                for src_i in range(uwb_num):
                    for src_j in range(uwb_num):
                        if src_i == src_j:
                            continue
                        rd = rd_calculate_withnoise(
                            src_us_global[src_i],
                            src_us_global[src_j],
                            if_nloss=False, if_senser_error=if_uwberror_list[src_i]
                        )

                        if test_output:
                            NodeFrameList[src_i][botid *
                                                    uwb_num + src_j].dis = rd
                        LinktrackNodeframeList[src_i].nodes[botid *
                                                            uwb_num + src_j].dis = rd
                continue
            # 其他bot的距离数据生成
            rec_us_global = uwb_position_transform(
                np.array(uwb_position),
                np.array(resample_rec_data[botid][0:3, i]),  # botid 0,1,2
                SO3(
                    quaternion=[resample_rec_data[botid][3, i], resample_rec_data[botid]
                                [4, i], resample_rec_data[botid][5, i], resample_rec_data[botid][6, i]]
                ).matrix()
            )
            # print("src_us_global:", src_us_global)
            # print("rec_us_global:", rec_us_global)
            # exit()
            # 计算相对距离,相对距离id以目标机id为准
            for src_uid in range(uwb_num):
                for rec_uid in range(uwb_num):
                    # 结算是否出现 bot间的多径效应
                    if_inter_uwb = np.random.uniform(0, 1) < nloss_error_inter_uwb
                    if_nloss = if_inter_bot or if_inter_uwb
                    rd = rd_calculate_withnoise(
                        src_us_global[src_uid],
                        rec_us_global[rec_uid],
                        if_nloss=if_nloss,
                        if_senser_error=if_uwberror_list[src_uid]
                    )

                    if test_output:
                        NodeFrameList[src_uid][botid *
                                                uwb_num + rec_uid].dis = rd
                    LinktrackNodeframeList[src_uid].nodes[botid *
                                                            uwb_num + rec_uid].dis = rd

        if test_output:
            NodeFrameTimeList.append(NodeFrameList)
        LinktrackNodeframeTimeList.append(LinktrackNodeframeList)

    if test_output:
        # 生成uwb_num个csv 分别输出
        from decimal import Decimal
        for src_uid in range(uwb_num):
            data = []
            for i in range(len(publish_time_list)):
                time = publish_time_list[i]
                di = [f"{Decimal(str(time)):f}"]
                for node in NodeFrameTimeList[i][src_uid]:
                    di.append(node.dis)

                data.append(di)

            # print("botnum * uwbnum" , botnum * uwb_num)
            # print("nodenum", len(NodeFrameTimeList[0][src_uid]))
            df = pd.DataFrame(data, columns=["stamp"] + [f"bot-{i//uwb_num}_{i%uwb_num}" for i in range(botnum * uwb_num)])
            df.to_csv(f"{src_name}_{src_uid}.csv", index=False)

    if bag_path == "":
        print("bag_path is empty, exit")
        return
    # 写入bag文件
    print("----读取bag文件:{}----".format(bag_path))
    with rosbag.Bag(bag_path, mode=mode) as bag:
        print("----写入bag文件:{}----".format(bag_path))

        for i in tqdm.tqdm(range(len(publish_time_list))):
            time = publish_time_list[i] * 1e-9
            LinktrackNodeframeL = LinktrackNodeframeTimeList[i]
            # 遍历所有的本地uwb
            for local_uid in range(uwb_num):
                Lframe = LinktrackNodeframeL[local_uid]
                Lframe: LinktrackNodeframe2
                # 设置时间戳
                stamp = rospy.Time.from_sec(time)
                tpc = topic_fomat.format(
                    src_name=src_name, local_uwb_id=local_uid)

                bag.write(tpc, Lframe, stamp)



for src_path_ in gt_path_list:
    src_name_ = src_path_.split("/")[-1].split("_gt_odom")[0]
    bag_path_ = bag_format_in.format(src_name=src_name_)
    rec_path_list_ = gt_path_list.copy()
    rec_path_list_.remove(src_path_)
    print("src_path:", src_path_)
    print("rec_path_list:", rec_path_list_)
    print("bag_path:", bag_path_)
    bag_path_ =""

    process_once(src_path_, rec_path_list_, topic_format_in,
                 bag_path_, uwb_position, publish_feq, noise_effect)
