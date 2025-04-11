import numpy as np
import scipy.signal as signal
from shapely.geometry import box


def counting_assist(img_depth,box_rgb):  #box [x1,y1,x2,y2] 左上，右下  img_depth(640,480)
    #step0 RGB（1280×720）box对应DEPTH（640×480）box  RGB[60:500,273:860]*1.09 
    count_num = 0
    x1_d=int((box_rgb[0]-395)*0.72)
    y1_d=int((box_rgb[1]-209)*0.72)
    x2_d=int((box_rgb[2]-395)*0.72)
    y2_d=int((box_rgb[3]-209)*0.72)

    rect1 = box(x1_d, y2_d, x2_d, y1_d)  #左下，右上
    rect2 = box(0, 480, 640, 0)
    intersection = rect1.intersection(rect2)  #计算相交区域
    # print('intersection:',intersection)
    if not intersection.is_empty:  #POLYGON EMPTY
        (x1_cross, y1_cross, x2_cross, y2_cross)=intersection.bounds  #左下，右上
        # print('(x1_cross, y1_cross, x2_cross, y2_cross):',(x1_cross, y1_cross, x2_cross, y2_cross))

        # xc_d=int((x1_cross+x2_cross)//2)    

        #获得depth中box对应区域的中心线area
        # area=img_depth[int(y1_cross):int(y2_cross),xc_d-10:xc_d+10]
        # area=1200-area  #曲线倒转 ()
        # print(img_depth.shape)
        # print(area.shape)
        mean=img_depth[int(y1_cross):int(y2_cross),int(x1_cross):int(x2_cross)]
        mean=1200-mean #曲线倒转 ()
        mean=mean[mean<740]
        t=np.mean(mean[500<mean])  #过线阈值
        # print('t:',t)

        # fish_line=np.average(area, axis=1) #求平均 一维数组
        # fish_line=fish_line[fish_line<700]  #去除噪点（550以上视为突变噪点，可调）
        # fish_line_nan=signal.medfilt(fish_line,45)  #中值滤波 窗口大小45,45可调
        # fish_line=fish_line_nan[~np.isnan(fish_line_nan)]
        count_num=0
        if t>545:
            count_num+=2
            # print('t>580')
        

        # if fish_line.shape[0]==0:
        #     return False

        # #获得超过阈值t*1.02的数组over_threshold_new
        # mask = np.concatenate(([False], fish_line > t*0.98, [False] ))
        # idx = np.flatnonzero(mask[1:] != mask[:-1])
        # over_threshold=[fish_line[idx[i]:idx[i+1]] for i in range(0,len(idx),2)]
        # over_threshold_new=[num for num in over_threshold if 120 < len(num)]
        # # print('over_threshold_new:',over_threshold_new)

        # count_num=0

        # #计算有几个超过阈值的数组（鱼数）
        # for i in over_threshold_new:
        #     # print(i)
        #     # print('len(i):',len(i),'(x2_cross-x1_cross)*0.75:',(x2_cross-x1_cross)*0.75)
        #     if len(i)>=(x2_cross-x1_cross)*0.75:
        #         # print(1)
        #         count_num=len(i)//310
        #     if len(i)<(x2_cross-x1_cross)*0.75:
        #         num=len([m for m in i if m > 430])
        #         if num<=15:
        #             # print(2)
        #             count_num+=1
        #         if num>15:
        #             # print(3)
        #             count_num+=2
                

    return count_num > 1
