"""robot_car_lidar.py
Webots/projects/vehicle/controllers/autonomous_vehile/autnomous_vehicle.cを参考にしています．
"""

import cv2
import math
import numpy as np
from vehicle import Driver, Car
from controller import GPS, Node

""" ロボットカークラス """
class RobotCar():
    SPEED   = 60        # 巡行速度
    UNKNOWN = 99999.99  # 黄色ラインを検出できない時の値
    FILTER_SIZE = 3     # 黄色ライン用のフィルタ
    TIME_STEP   = 30    # センサのサンプリング時間[ms]
    CAR_WIDTH   = 2.015 # 車幅[m]
    CAR_LENGTH  = 5.0   # 車長[m]
    k_sp = 200          # ステアリング角度に対する速度の変化係数
    ave_list = [0,0,0]  # 移動平均フィルタの初期値
    list_value = 3      # 移動平均フィルタのリスト数
    list_sum = 0    # 移動平均フィルタの合計値
    diff_steer_angle = 0.0    # ステアリング角度の差
    v_x = 0.0           # x方向の速度
    v_y = 0.0           # y方向の速度
    phi = 0.0           # ステアリングの角度
    pos_x = 0.0           # x座標
    pos_y = 0.0           # y座標
    theta = 0.0           # 角度
        
    """ コンストラクタ """
    def __init__(self):  
        # Welcome message
        self.welcomeMessage()    
      
        # Driver
        self.driver = Driver()
        self.driver.setSteeringAngle(0)
        self.driver.getSteeringAngle() # ステアリング角の取得
        self.driver.getCurrentSpeed() # フォグランプ点灯
        self.driver.setCruisingSpeed(self.SPEED) 
        self.driver.setDippedBeams(True) # ヘッドライト点灯

        # Car
        self.car = Car()
        self.car.getWheelbase() # ホイールベース[m]
        self.car.getRearWheelRadius() # 車輪の半径
 
        
        # LIDAR (SICK LMS 291)
        self.lidar = self.driver.getDevice("Sick LMS 291")
        self.lidar.enable(self.TIME_STEP)
        self.lidar_width = self.lidar.getHorizontalResolution()
        self.lidar_range = self.lidar.getMaxRange()
        self.lidar_fov   = self.lidar.getFov()
        
        # GPS
        self.gps = self.driver.getDevice("gps")
        self.gps.enable(self.TIME_STEP)

        # Camera
        self.camera = self.driver.getDevice("camera")
        self.camera.enable(self.TIME_STEP)
        self.camera_width  = self.camera.getWidth()
        self.camera_height = self.camera.getHeight()
        self.camera_fov    = self.camera.getFov()
        print("camera: width=%d height=%d fov=%g" % \
            (self.camera_width, self.camera_height, self.camera_fov))

        # Display
        self.display = self.driver.getDevice("display")
        self.display.attachCamera(self.camera) # show camera image
        self.display.setColor(0xFF0000)
        self.display.setFont('Arial', 12, True) 
        
      
    """ ウエルカムメッセージ """    
    def welcomeMessage(self):
        print("*********************************************")
        print("*   Welcome to a simple robot car program   *")
        print("*********************************************")       
                

    """ 移動平均フィルタを実装しよう．"""
    def maFilter(self, new_value):
        """
        ここに移動平均フィルタのコードを書く
        """   
        self.list_sum = 0
        if new_value == self.UNKNOWN:
            return self.UNKNOWN
        for i in range(self.list_value-1):
            self.ave_list[i] = self.ave_list[i+1]
            self.list_sum += self.ave_list[i]
        self.ave_list[self.list_value-1] = new_value
        new_value = (self.ave_list[0] + self.list_sum) / self.list_value
        return new_value

    
    """ ステアリングの制御： wheel_angleが正だと右折，負だと左折 """
    def control(self, steering_angle):
        LIMIT_ANGLE = 0.5 # ステアリング角の限界 [rad]
        if steering_angle > LIMIT_ANGLE:
            steering_angle = LIMIT_ANGLE
        elif steering_angle < -LIMIT_ANGLE:
            steering_angle = -LIMIT_ANGLE
        self.driver.setSteeringAngle(steering_angle)
    
    
    """ 画素と黄色の差の平均を計算 """
    def colorDiff(self, pixel, yellow):
        d, diff = 0, 0
        for i in range (0,3):
            d = abs(pixel[i] - yellow[i])
            diff += d 
        return diff/3


    """ 黄色ラインを追従するための操舵角の計算　"""
    def calcSteeringAngle(self,image):
        YELLOW = [95, 187, 203]   # 黄色の値 (BGR フォーマット)
        sumx = 0                  # 黄色ピクセルのX座標の和
        sumy = 0
        pixel_count = 0           # 黄色の総ピクセル数
        for y in range(int(1.0*self.camera_height/3.0),self.camera_height):
            for x in range(0,self.camera_width):
                if self.colorDiff(image[y,x], YELLOW) < 30: # 黄色の範囲
                    sumx += x                      # 黄色ピクセルのx座標の和
                    sumy += y
                    pixel_count += 1               # 黄色ピクセル数の和
   
        # 黄色ピクセルを検出できない時は no pixels was detected...

        if pixel_count == 0:
            return self.UNKNOWN
        else:
            x_ave = float(sumx) /(pixel_count * self.camera_width)
            y_ave = float(sumy) /(pixel_count * self.camera_height)
            steer_angle = (x_ave - 0.28) * self.camera_fov
            # if sumy > 3700 and sumx > 2500 and pixel_count > 100 or sumx < 1000:
            if x_ave > 0.35:
                steer_angle = 0
            return steer_angle

    """ 障害物の方位と距離を返す. 障害物を発見できないときはUNKNOWNを返す．
        ロボットカー正面の矩形領域に障害物がある検出する    
    """
    def calcObstacleAngleDist(self, lidar_data):
        OBSTACLE_HALF_ANGLE = 20.0 # 障害物検出の角度．中央の左右20[°]．
        OBSTACLE_DIST_MAX   = 20.0 # 障害物検出の最大距離[m]
        OBSTACLE_MARGIN     = 0.1  # 障害物回避時の横方向の余裕[m]
        # self.lidar_width 水平方向の分解能は180
        
        sumx = 0
        collision_count = 0
        obstacle_dist   = 0.0
        for i in range(int(self.lidar_width/2 - OBSTACLE_HALF_ANGLE), \
            int(self.lidar_width/2 + OBSTACLE_HALF_ANGLE)):
            dist = lidar_data[i]
            if dist < OBSTACLE_DIST_MAX: 
                sumx += i 
                collision_count += 1 
                obstacle_dist += dist 
                
        if collision_count == 0 or obstacle_dist > collision_count * OBSTACLE_DIST_MAX:  # 障害物を検出できない場合
            return self.UNKNOWN, self.UNKNOWN
        else:
            obstacle_angle = (float(sumx) /(collision_count * self.lidar_width) - 0.5) * self.lidar_fov
            obstacle_dist  = obstacle_dist/collision_count
            
            if abs(obstacle_dist * math.sin(obstacle_angle)) < 0.5 * self.CAR_WIDTH + OBSTACLE_MARGIN:
                return obstacle_angle, obstacle_dist # 衝突するとき
            else:
                return self.UNKNOWN, self.UNKNOWN

    """
    Dead Reckoningの計算
    """
    def DeadReckoning(self, mps, phi):
        dot_theta = math.tan(phi) * mps / self.car.getWheelbase() # 角度の変化
        self.theta += dot_theta * self.TIME_STEP / 1000 # 車体の角度 ステップ時間で積分する
        self.v_x = mps * math.cos(self.theta) # x方向の速度
        self.v_y = mps * math.sin(self.theta) # y方向の速度
        # print("velocity:%.2f, %.2f, %.2f " % (self.v_x, self.v_y, dot_theta), end='')
        self.pos_x += self.v_x * self.TIME_STEP / 1000 # 車体のx座標 ステップ時間で積分する
        self.pos_y += self.v_y * self.TIME_STEP / 1000 # 車体のy座標 ステップ時間で積分する
        return self.pos_x, self.pos_y, self.theta

    """ レンジイメージの表示 """
    def LidarMapping(self, lidar_data):
        self.display.setAlpha(0.0)
        self.display.fillRectangle(10,60, 180, 80)

        self.display.setAlpha(0.5)
        for i in range(self.lidar_width):
            if lidar_data[i] == float('inf'):
                lidar_data[i] = self.lidar_range
            # self.display.drawLine(10+i, 150-10, 10+i, 140-int(lidar_data[i])) # レンジイメージの表示
            self.display.drawLine(100, 150-10, 
                                  100-int(lidar_data[i]*math.cos(i*math.pi/180)), 
                                    150-10-int(lidar_data[i]*math.sin(i*math.pi/180))) # レンジイメージの表示

    """ 実行 """
    def run(self): 
        stop =  False
        step = 0       
        init_flag = True
        print("time,dead_x,dead_y,dead_theta,gps_x, gps_y,gps_z,error_x,error_y")

        while self.driver.step() != -1:
            step +=1 
            BASIC_TIME_STEP = 10 # シミュレーションの更新は10[ms]

            if stop == True:
                print("Stop!")
                self.driver.setCruisingSpeed(0)

            # センサの更新をTIME_STEP[ms]にする．この例では60[ms] 
            if step % int(self.TIME_STEP / BASIC_TIME_STEP) == 0:
                print("%d," % step, end='')

                # Lidar
                lidar_data = self.lidar.getRangeImage()
                # self.LidarMapping(lidar_data)

                # GPS
                gps_values = self.gps.getValues()
                if init_flag:
                    init_values = gps_values
                    init_flag = False
                else:
                    gps_values[0] = gps_values[0] - init_values[0]
                    gps_values[1] = gps_values[1] - init_values[1]
                    gps_values[2] = gps_values[2] - init_values[2] 
                mps = self.gps.getSpeed()
                kmph = self.driver.getCurrentSpeed() # 内界センサによる車の速度

                dead_values = self.DeadReckoning(kmph * 1000 / 3600 , self.driver.getSteeringAngle()) # デッドレコニングの計算

                # print()関数を用いてデッドレコニングとGPS，誤差の数値を出力する
                print("%f,%f,%f," % (dead_values[0], dead_values[1], dead_values[2]), end='')
                print("%f,%f,%f," % (gps_values[0], gps_values[1], gps_values[2]), end='')
                print("%f,%f" % ((dead_values[0] + gps_values[0])/(-gps_values[0]), (dead_values[1] - gps_values[1])/gps_values[1]))

                # Camera
                camera_image = self.camera.getImage()
                # OpenCVのPythonは画像をnumpy配列で扱うので変換する．            
                cv_image = np.frombuffer(camera_image, np.uint8).reshape((self.camera_height, self.camera_width, 4))

                # 障害物回避
                obstacle_angle, obstacle_dist = self.calcObstacleAngleDist(lidar_data)
                STOP_DIST = 7 # 停止距離[m]80mまで
                
                if obstacle_dist < STOP_DIST:
                    # print("%d:Find obstacles(angle=%g, dist=%g)" % (step, obstacle_angle, obstacle_dist))
                    # stop = True
                    self.driver.setSteeringAngle(math.pi/6) # 障害物を検出したとき指定角度にステアリングを切る

                else:                
                    # 操作量（ステアリング角度）の計算
                    steering_angle = self.maFilter(self.calcSteeringAngle(cv_image))
                    controlled_speed = float(self.SPEED - self.k_sp * abs(steering_angle)) # カーブでの減速制御 

                    if steering_angle != self.UNKNOWN:
                        self.driver.setCruisingSpeed(controlled_speed)  # 走行速度の設定
                        self.control(steering_angle)              # ステアリングの制御(比例制御の簡易版)
                        stop = False
                    else: 
                        self.driver.setCruisingSpeed(0) # 追従する黄色線をロストしたので停止する．
                        # stop = True

""" メイン関数 """ 
def main():  
    robot_car = RobotCar() 
    robot_car.run()  
    
""" このスクリプトをモジュールとして使うための表記 """
if __name__ == '__main__':
    main()