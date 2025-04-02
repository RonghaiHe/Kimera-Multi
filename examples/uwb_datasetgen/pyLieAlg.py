# py实现李代数基本运算

import numpy as np
from typing import Union

class SO3:
    '''
        SO3 群
    '''

    def __init__(self, * , matrix:Union[np.ndarray, list]=None,
                    quaternion:Union[np.ndarray, list]=None, Euler:Union[np.ndarray, list]=None):
        '''
            SO3 群构造
            @param matrix: 3x3 旋转矩阵
            @param quaternion: 4x1 四元数
            @param Euler: 欧拉角 Z-Y'-Z''
        '''

        # 判断输入是否不止一种，否则报错
        assert + (matrix is not None) + (quaternion is not None) + (Euler is not None) == 1, '仅允许向量、矩阵、四元数或欧拉角其中之一作为输入'

        if isinstance(matrix, list):
            assert len(matrix) == 3 and len(matrix[0]) == 3, '必须为3x3矩阵'
            self.w = np.array(matrix)
        elif isinstance(matrix, np.ndarray):
            assert matrix.shape == (3, 3), '必须为3x3矩阵'
            self.w = matrix
        
        if isinstance(quaternion, list):
            assert len(quaternion) == 4, '必须为4x1四元数'
            quaternion = np.array(quaternion)
            self.w = self.Quaternion2Matrix(quaternion)
        elif isinstance(quaternion, np.ndarray):
            assert quaternion.shape == (4,), '必须为4x1四元数'
            self.w = self.Quaternion2Matrix(quaternion)

        if isinstance(Euler, list):
            assert len(Euler) == 3, '必须为欧拉角'
            Euler = np.array(Euler)
            self.w = self.Euler2Matrix(Euler)
        elif isinstance(Euler, np.ndarray):
            assert Euler.shape == (3,), '必须为欧拉角'
            self.w = self.Euler2Matrix(Euler)

    def __mul__(self, other):
        '''
            旋转矩阵乘法
        '''
        assert isinstance(other, SO3), '只能与SO3相乘'
        return SO3(matrix=self.w @ other.w)
    
    def inverse(self):
        '''
            旋转矩阵求逆
        '''
        return SO3(matrix=self.w.T)


    # SO3群到李代数的映射: 对数映射
    def log(self, logmethod = False):
        '''
            SO3到李代数的映射
            @param logmethod: 如果为false 采用 Ra = a 求解转轴，theta = arccos((Tr(R)-1)/2); so3 = theta * a
            如果为true ，采用 theta^ = log(R) = theta / 2 sin(theta) * (R - R.T)
        '''
        err = 1e-6
        if np.linalg.norm(np.eye(3) - self.w) < err:
            # case theta == 0
            return so3(vector=np.array([0, 0, 0]))
        elif np.abs(np.trace(self.w) + 1) < err:
            # case theta == pi
            if self.w[0, 0] > 0:
                w = 1 / np.sqrt(2 * (1 + self.w[0, 0])) * np.array([1 + self.w[0, 0], self.w[1, 0], self.w[2, 0]])  # wx
            elif self.w[1, 1] > err:
                w = 1 / np.sqrt(2 * (1 + self.w[1, 1])) * np.array([self.w[0, 1], 1 + self.w[1, 1], self.w[2, 1]])  # wy
            else:
                w = 1 / np.sqrt(2 * (1 + self.w[2, 2])) * np.array([self.w[0, 2], self.w[1, 2], 1 + self.w[2, 2]])  # wz
            w = np.pi * w
            return so3(vector=w)

        if logmethod:
            # theta^ = log(R) = theta / 2 sin(theta) * (R - R.T)
            theta = np.arccos((np.trace(self.w) - 1) / 2)
            w = theta / (2 * np.sin(theta)) * (self.w - self.w.T) 
            return so3(vector=so3.so3vee(w))
        else:
            # Ra = a 求解转轴，theta = arccos((Tr(R)-1)/2); so3 = theta * a
            theta = np.arccos((np.trace(self.w) - 1) / 2)
            if np.abs(theta) < err:
                return so3(vector=np.array([0, 0, 0]))
            a = 1 / (2 * np.sin(theta)) * np.array([self.w[2, 1] - self.w[1, 2], self.w[0, 2] - self.w[2, 0], self.w[1, 0] - self.w[0, 1]]).reshape(3,)
            return so3(vector=theta * a)

        
    def quaternion(self):
        '''
            返回四元数
        '''
        return self.Matrix2Quaternion(self.w)
    
    def euler(self):
        '''
            返回欧拉角
        '''
        return self.Matrix2Euler(self.w)
    def matrix(self):
        '''
            返回旋转矩阵
        '''
        return self.w

    @staticmethod
    def Euler2Matrix(Euler:Union[np.ndarray, list]):
        '''
            从欧拉角构造旋转矩阵
            @param Euler: 欧拉角
        '''
        assert len(Euler) == 3, '必须为欧拉角'

        Rx = np.array([[1, 0, 0],
                          [0, np.cos(Euler[0]), -np.sin(Euler[0])],
                          [0, np.sin(Euler[0]), np.cos(Euler[0])]]) 
        Ry = np.array([[np.cos(Euler[1]), 0, np.sin(Euler[1])],
                          [0, 1, 0],
                          [-np.sin(Euler[1]), 0, np.cos(Euler[1])]])
        Rz = np.array([[np.cos(Euler[2]), -np.sin(Euler[2]), 0],
                          [np.sin(Euler[2]), np.cos(Euler[2]), 0],
                          [0, 0, 1]])
        R = Rz @ Ry @ Rx
        return R
    
    @staticmethod
    def Quaternion2Matrix(quaternion:Union[np.ndarray, list]):
        '''
            从四元数构造旋转矩阵
            @param quaternion: 四元数
        '''
        assert len(quaternion) == 4, '必须为四元数'
        q0, q1, q2, q3 = quaternion
        R = np.array([[q0**2 + q1**2 - q2**2 - q3**2, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
                      [2*(q1*q2 + q0*q3), q0**2 - q1**2 + q2**2 - q3**2, 2*(q2*q3 - q0*q1)],
                      [2*(q1*q3 - q0*q2), 2*(q0*q1 + q2*q3), q0**2 - q1**2 - q2**2 + q3**2]])
        return R
    
    @staticmethod
    def Matrix2Quaternion(matrix:Union[np.ndarray, list]):
        '''
            从旋转矩阵构造四元数
            @param matrix: 3x3 旋转矩阵
        '''
        assert len(matrix) == 3 and len(matrix[0]) == 3, '必须为3x3矩阵'
        q = np.zeros((4,))
        t = np.trace(matrix)
        if t > 0:
            s = np.sqrt(t + 1.0) * 2
            q[0] = 0.25 * s
            q[1] = (matrix[2][1] - matrix[1][2]) / s
            q[2] = (matrix[0][2] - matrix[2][0]) / s
            q[3] = (matrix[1][0] - matrix[0][1]) / s
        else:
            if (matrix[0][0] > matrix[1][1]) and (matrix[0][0] > matrix[2][2]):
                s = np.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2
                q[1] = 0.25 * s
                q[2] = (matrix[0][1] + matrix[1][0]) / s
                q[3] = (matrix[0][2] + matrix[2][0]) / s
                q[0] = (matrix[2][1] - matrix[1][2]) / s
            elif (matrix[1][1] > matrix[2][2]):
                s = np.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2
                q[2] = 0.25 * s
                q[3] = (matrix[1][2] + matrix[2][1]) / s
                q[0] = (matrix[0][1] - matrix[1][0]) / s
                q[1] = (matrix[0][2] - matrix[2][0]) / s
            else:
                s = np.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2
                q[3] = 0.25 * s
                q[0] = (matrix[1][0] - matrix[0][1]) / s
                q[1] = (matrix[0][2] + matrix[2][0]) / s
                q[2] = (matrix[1][2] + matrix[2][1]) / s
        return q
    

    

class so3:
    '''
        so3 李代数
    '''
    def __init__(self, * ,vector:Union[np.ndarray, list], matrix:Union[np.ndarray, list]=None,
                    quaternion:Union[np.ndarray, list]=None, Euler:Union[np.ndarray, list]=None):
        '''
            so3 李代数构造
            @param vector: ω 旋转3维向量
            @param matrix: 3x3 旋转矩阵
            @param quaternion: 4x1 四元数
            @param Euler: 欧拉角
        '''

        # 判断输入是否不止一种，否则报错
        assert (vector is not None) + (matrix is not None) + (quaternion is not None) + (Euler is not None) == 1, '仅允许向量、矩阵、四元数或欧拉角其中之一作为输入'
        # 判断输入是否为3x1向量
        if isinstance(vector, list):
            assert len(vector) == 3, '必须为3x1向量'    
            self.w = np.array(vector)
        elif isinstance(vector, np.ndarray):
            assert vector.shape == (3,), '必须为3x1向量'
            self.w = vector

        if isinstance(matrix, list):
            assert len(matrix) == 3 and len(matrix[0]) == 3, '必须为3x3矩阵'
            self.w = self.Matrix2so3alg(np.array(matrix))
        elif isinstance(matrix, np.ndarray):
            assert matrix.shape == (3, 3), '必须为3x3矩阵'
            self.w = self.Matrix2so3alg(matrix)

        if isinstance(quaternion, list):
            assert len(quaternion) == 4, '必须为4x1四元数'
            quaternion = np.array(quaternion)
            self.w = self.Quaternion2so3alg(quaternion)
        elif isinstance(quaternion, np.ndarray):
            assert quaternion.shape == (4,), '必须为4x1四元数'
            self.w = self.Quaternion2so3alg(quaternion)

        if isinstance(Euler, list):
            assert len(Euler) == 3, '必须为欧拉角'
            Euler = np.array(Euler)
            self.w = self.Euler2so3alg(Euler)
        elif isinstance(Euler, np.ndarray):
            assert Euler.shape == (3,), '必须为欧拉角'
            self.w = self.Euler2so3alg(Euler)

    def exp(self):
        '''
            李代数到SO3的映射: 指数映射
        '''
        assert self.w.shape == (3,), '必须为3x1李代数'
        theta = np.linalg.norm(self.w)
        if theta == 0:
            return SO3(matrix=np.eye(3))
        else:
            w = self.w / theta
            w_skew = self.so3hat(w)
            return SO3(matrix=np.eye(3) + np.sin(theta) * w_skew + (1 - np.cos(theta)) * w_skew @ w_skew)

    # def __add__(self, other):
    #     '''
    #         李代数加法
    #     '''
    #     assert isinstance(other, so3), '只能与so3相加'
    #     R1 = self.exp()
    #     R2 = other.exp()
    #     R = R1 * R2
    #     return R.log()
    
    def vector(self):
        '''
            返回李代数向量
        '''
        return self.w
    def matrix(self):
        '''
            返回李代数矩阵
        '''
        return self.so3hat(self.w)
    
    def magnitude(self):
        '''
            返回李代数模,即角度
        '''
        return np.linalg.norm(self.w)
    
    def numbermultiply(self, other):
        '''
            李代数数乘
        '''
        assert isinstance(other, (int, float)), '只能与数字相乘'
        return so3(vector=self.w * other)

    @staticmethod
    def so3hat(vector:Union[np.ndarray, list]):
        '''
            反对称化
            @param vector: 3x1 向量
        '''
        assert len(vector) == 3, '必须为3x1向量'
        return np.array([[0, -vector[2], vector[1]],
                         [vector[2], 0, -vector[0]],
                         [-vector[1], vector[0], 0]])
    @staticmethod
    def so3vee(matrix:Union[np.ndarray, list]):
        '''
            反对称化
            @param matrix: 3x3 矩阵
        '''
        assert len(matrix) == 3 and len(matrix[0]) == 3, '必须为3x3矩阵'
        return np.array([matrix[2][1], matrix[0][2], matrix[1][0]]).reshape(3,)

    @staticmethod
    def Matrix2so3alg(matrix:Union[np.ndarray, list]):
        '''
            从旋转矩阵构造李代数
            @param matrix: 3x3 旋转矩阵
        '''
        assert len(matrix) == 3 and len(matrix[0]) == 3, '必须为3x3矩阵'
        return SO3(matrix=matrix).log()
    
    @staticmethod
    def Quaternion2so3alg(quaternion:Union[np.ndarray, list]):
        '''
            从四元数构造李代数
            @param quaternion: 4x1 四元数
        '''
        assert len(quaternion) == 4, '必须为4x1四元数'
        return SO3(quaternion=quaternion).log()
    
    @staticmethod
    def Euler2so3alg(Euler:Union[np.ndarray, list]):
        '''
            从欧拉角构造李代数
            @param Euler: 欧拉角
        '''
        assert len(Euler) == 3, '必须为欧拉角'
        return SO3(Euler=Euler).log()
    
    @staticmethod
    def so3alg2Matrix(vector:Union[np.ndarray, list]):
        '''
            从李代数构造旋转矩阵
            @param vector: 3x1 李代数
        '''
        assert len(vector) == 3, '必须为3x1李代数'
        return SO3(vector=vector).log()
    
    @staticmethod
    def so3alg2Quaternion(vector:Union[np.ndarray, list]):
        '''
            从李代数构造四元数
            @param vector: 3x1 李代数
        '''
        assert len(vector) == 3, '必须为3x1李代数'
        return SO3(vector=vector).log()


class SE3:
    '''
        SE3 群
    '''
    def __init__(self, * , matrix:Union[np.ndarray, list]=None,tuple:tuple=None):
        '''
            SE3 群构造
            @param matrix: 4x4 变换矩阵
            @param tuple: (平移向量，旋转矩阵)
        '''

        # 判断输入是否不止一种，否则报错
        assert (matrix is not None) + (tuple is not None) == 1, '仅允许矩阵、四元数或欧拉角其中之一作为输入'

        if isinstance(matrix, list):
            assert len(matrix) == 4 and len(matrix[0]) == 4, '必须为4x4矩阵'
            self.w = np.array(matrix)
        elif isinstance(matrix, np.ndarray):
            assert matrix.shape == (4, 4), '必须为4x4矩阵'
            self.w = matrix

        if isinstance(tuple, tuple):
            assert len(tuple) == 2, '必须为平移向量和旋转向量'
            self.w = self.RtVector2SE3(tuple[0], tuple[1])

    def __mul__(self, other):
        '''
            变换矩阵乘法
        '''
        assert isinstance(other, SE3), '只能与SE3相乘'
        return SE3(matrix=self.w @ other.w)
    
    
    
    def log(self):
        '''
            SE3到se3李代数的映射
        '''
        R = self.w[:3, :3]
        t = self.w[:3, 3]
        w = SO3(matrix=R).log()

        I = np.eye(3)
        theta = w.magnitude()
        wx = w.matrix()
        wx2 = wx @ wx
        a = (1 / theta**2) * (1 - (theta * np.sin(theta)) / (2 * (1 - np.cos(theta)))) if theta != 0 else 1 / 12
        V = I - 1 / 2.0 * wx + a * wx2

        v = V.dot(t)

        return se3(vector=np.append(w.vector(), v))
    
    def matrix(self):
        '''
            返回变换矩阵
        '''
        return self.w


    @staticmethod
    def RtVector2SE3(transform:Union[np.ndarray, list],Rotation:Union[np.ndarray, list]):
        '''
            从平移向量和旋转矩阵构造变换矩阵
            @param transform: 平移向量
            @param Rotation: 旋转矩阵
        '''
        assert len(transform) == 3, '必须为平移向量'
        assert len(Rotation) == 3 and len(Rotation[0]) == 3, '必须为旋转矩阵'
        T = np.eye(4)
        T[:3, :3] = Rotation
        T[:3, 3] = transform
        return T

class se3:
    '''
        se3 李代数
    '''

    def __init__(self, vector:Union[np.ndarray, list], matrix:Union[np.ndarray, list]=None,
                    tuple:tuple=None):
        '''
            se3 李代数构造
            @param vector: [ ρ， ω] ρ为平移3维平移， ω为so3李代数
            @param matrix: 4x4 变换矩阵
            @param tuple: (平移向量，旋转矩阵)

        '''

        # 判断输入是否不止一种，否则报错
        assert (vector is not None) + (matrix is not None) + (tuple is not None) == 1, '仅允许向量、矩阵或平移向量和旋转表示其中之一作为输入'

        # 判断输入是否为6x1向量
        if isinstance(vector, list):
            assert len(vector) == 6, '必须为6x1向量'
            self.w = np.array(vector)
        elif isinstance(vector, np.ndarray):
            assert vector.shape == (6, 1), '必须为6x1向量'
            self.w = vector

        if isinstance(matrix, list):
            assert len(matrix) == 4 and len(matrix[0]) == 4, '必须为4x4矩阵'
            self.w = self.Matrix2se3alg(np.array(matrix))
        elif isinstance(matrix, np.ndarray):
            assert matrix.shape == (4, 4), '必须为4x4矩阵'
            self.w = self.Matrix2se3alg(matrix)
        if isinstance(tuple, tuple):
            assert len(tuple) == 2, '必须为平移向量和旋转表示'
            r = tuple[1]
            if isinstance(r, np.ndarray):
                assert r.shape == (3, 3), 'tuple中的旋转矩阵必须为3x3'
            elif isinstance(r, list):
                assert len(r) == 3 and len(r[0]) == 3, 'tuple中的旋转矩阵必须为3x3'

    def vector(self):
        '''
            返回李代数向量
        '''
        return self.w
    
    def matrix(self):
        '''
            返回李代数矩阵
        '''
        return self.se3hat(self.w)
    
    def exp(self):
        '''
            se3到SE3的映射: 指数映射
        '''
        w = so3(vector=self.w[0:3])
        R = w.exp().matrix()
        t = self.w[3:6]

        theta = w.magnitude()
        I = np.eye(3)
        wx = w.matrix()
        wx2 = wx.dot(wx)
        A = (1 - np.cos(theta)) / (theta ** 2) if theta != 0 else 1 / 2.0
        B = (theta - np.sin(theta)) / (theta ** 3) if theta != 0 else 1 / 6.0
        V = I + A * wx + B * wx2
        t = V.dot(t)

        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = t
        return SE3(matrix=T)
    
    @staticmethod
    def se3hat(vector:Union[np.ndarray, list]):
        '''
            反对称化
            @param vector: 6x1 向量
        '''
        assert len(vector) == 6, '必须为6x1向量'
        w = vector[:3]
        v = vector[3:]
        return np.array([[0, -w[2], w[1], v[0]],
                         [w[2], 0, -w[0], v[1]],
                         [-w[1], w[0], 0, v[2]],
                         [0, 0, 0, 0]])
    
    @staticmethod
    def se3vee(matrix:Union[np.ndarray, list]):
        '''
            反对称化
            @param matrix: 4x4 矩阵
        '''
        assert len(matrix) == 4 and len(matrix[0]) == 4, '必须为4x4矩阵'
        return np.array([matrix[2][1], matrix[0][2], matrix[1][0], matrix[0][3], matrix[1][3], matrix[2][3]]).reshape(6, 1)

        

if __name__ == '__main__':
    # 测试代码
    roll = 30
    pitch = 45
    yaw = 60
    
    d = np.array([roll, pitch, yaw])
    print(d.shape)
    exit()

    def deg2rad(deg):
        return deg * np.pi / 180
    
    # 测试欧拉角
    SO3test = SO3(Euler=[deg2rad(roll), deg2rad(pitch), deg2rad(yaw)])
    print('SO3 群:', SO3test.matrix())
    print('SO3 群的四元数:', SO3test.quaternion())

    print("r.T * r = " + str(SO3test.matrix().T @ SO3test.matrix()))

    so3_1 = SO3test.log()
    so3_2 = SO3test.log(logmethod=True)

    print('SO3 群到李代数的映射:', so3_1.vector())
    print('SO3 群到李代数的映射:', so3_2.vector())