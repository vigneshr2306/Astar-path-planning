import cv2
import numpy as np
from matplotlib import pyplot as plt
import struct
import serial
import time

data = serial.Serial('COM10',115200,serial.EIGHTBITS)

#Processing the first image
cap = cv2.VideoCapture(0)
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        cv2.imshow('frame', frame)
        cv2.imwrite('img1.png',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
img = cv2.imread('img1.png')
img2hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lower_red = np.array([0, 100, 100])
upper_red = np.array([15, 255, 255])
mask = cv2.inRange(img2hsv, lower_red, upper_red)
res = cv2.bitwise_and(img,img,mask=mask)
cv2.destroyAllWindows()

#A* Algorithm starts
res = cv2.imread('config.png')

grid_size = 5
rstep = res.shape[0]/grid_size
cstep = res.shape[1]/grid_size


grid = []
grid1 = []
grid2 = []
h_val = []
g_val = []
f_val = []
temp = []
temp_f_val = []
temp_g_val = []
o_list = []
c_list = []
index = []
path = []

#Getting the 4 boundaries
corn1 = []
corn2 = []
corn3 = []
corn4 = []

for i in range(0,(grid_size*grid_size),grid_size):
    corn1.append(i)
for i in range(0,grid_size,1):
    corn2.append(i)
for i in range((grid_size-1),(grid_size*grid_size),grid_size):
    corn3.append(i)
for i in range((grid_size*(grid_size-1)),(grid_size*grid_size),1):
    corn4.append(i)

for i in range(0,res.shape[0],rstep):
    for j in range (0,res.shape[1],cstep):
        temp = res[i:i+rstep,j:j+cstep]
        temp2gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        grid.append(temp)
        grid1.append(temp2gray)

for i in range(0,(grid_size*grid_size),1):
        plt.subplot(grid_size,grid_size,i+1), plt.imshow(grid[i]), plt.title(i)
        plt.xticks([]), plt.yticks([])
plt.show()

maxi = []
for i in range(0,grid_size*grid_size,1):
    maxi.append(cv2.countNonZero(grid1[i]))
ind = maxi.index(max(maxi))
maxi.remove(max(maxi))
ind1 = maxi.index(max(maxi))
del maxi[:]

print ind,ind1
end_point1 = input("Enter end point1:")
end_point2 = input("Enter end point2:")

lower_blue = np.array([80, 80,100])
upper_blue = np.array([120, 255, 255])
masking = cv2.inRange(img2hsv, lower_blue, upper_blue)
obs = cv2.bitwise_and(img,img,mask=masking)
obs = cv2.medianBlur(obs, 5)
for i in range(0,obs.shape[0],rstep):
    for j in range (0,obs.shape[1],cstep):
        temp3 = obs[i:i+rstep,j:j+cstep]
        temp3gray = cv2.cvtColor(temp3, cv2.COLOR_BGR2GRAY)
        grid2.append(temp3gray)
for i in range(0,grid_size*grid_size,1):
    maxi.append(cv2.countNonZero(grid2[i]))
for i in range(0,grid_size*grid_size,1):
    if maxi[i]>50:
        c_list.append(i)

start_point = grid[ind]
end_point = grid[0]
current_position = start_point

z = [[0 for x in range(grid_size)]for y in range(grid_size)]
l=0
for i in range(0, grid_size, 1):
    for j in range(0, grid_size, 1):
        z[i][j] = l
        l = l + 1

def astar():
    p_list = [(-1) for x in range(grid_size * grid_size)]
    for k in range(0,(grid_size*grid_size),1):
        a = ind
        b = end_point1
        c1 = 0
        c2 = 0
        for i in range(0, grid_size, 1):
            for j in range(0, grid_size, 1):
                if z[i][j] == k:
                    row_index_val = i
                    col_index_val = j
                if z[i][j] == a:
                    row_index1 = i
                    col_index1 = j
                if z[i][j] == b:
                    row_index2 = i
                    col_index2 = j
        v1 = row_index1 - row_index_val
        v2 = row_index2 - row_index_val
        while(1):
            if (a-k) == 0:
                h_val.append(c1*10)
                break
            elif v1 == 0:
                if (a-k) > 0:
                    a = a-1
                    c1 = c1+1
                elif (a-k) < 0:
                    a = a+1
                    c1 = c1+1
            elif v1 > 0:
                a = a-((abs(v1))*grid_size)
                c1 = c1+(abs(v1))
                v1 = 0
            elif v1 < 0 :
                a = a+((abs(v1))*grid_size)
                c1 = c1+(abs(v1))
                v1 = 0
        while (1):
            if (b - k) == 0:
                g_val.append(c2 * 10)
                break
            elif v2 == 0:
                if (b - k) > 0:
                    b = b - 1
                    c2 = c2 + 1
                elif (b - k) < 0:
                    b = b + 1
                    c2 = c2 + 1
            elif v2 > 0:
                b = b - ((abs(v2)) * grid_size)
                c2 = c2 + (abs(v2))
                v2 = 0
            elif v2 < 0:
                b = b + ((abs(v2)) * grid_size)
                c2 = c2 + (abs(v2))
                v2 = 0
        f_val.append(h_val[k] + g_val[k])

    p = 1
    start_point_index = ind
    end_point_index = end_point1
    current_position_index = start_point_index

    while(1):
        if current_position_index == end_point_index:
            break
        else:
            c_list.append(current_position_index)
            k1=5
            k2=5
            k3=5
            k4=5
            a = current_position_index + 1
            b = current_position_index - 1
            c = current_position_index + grid_size
            d = current_position_index - grid_size

            for i in range(1,grid_size,1):
                if current_position_index == grid_size*i:
                    k1 = 0
                    k3 = 0
                    k4 = 0
                    i = 0
                    break
                elif current_position_index == ((grid_size*grid_size)-1) - (grid_size*i):
                    k2 = 0
                    k3 = 0
                    k4 = 0
                    i = 0
                    break
            if i!=0:
                k1 = 0
                k2 = 0
                k3 = 0
                k4 = 0
            for i in range(0, len(c_list), 1):
                if (a - c_list[i]) == 0:
                    k1 = 1
                if (b - c_list[i]) == 0:
                    k2 = 1
                if (c - c_list[i]) == 0:
                    k3 = 1
                if (d - c_list[i]) == 0:
                    k4 = 1

            temp_h_val1 = []
            a1 = []
            indi = []

            if (a <= ((grid_size*grid_size)-1)) & (a >= 0) & (k1 == 0):
                o_list.append(a)
                a1.append(a + 1)
                a1.append(a - 1)
                a1.append(a + grid_size)
                a1.append(a - grid_size)
                if (a1[0] >= 0) & (a1[0] <= ((grid_size*grid_size)-1)) & (a not in corn3) & (a1[0] not in c_list):
                    temp_h_val1.append(h_val[a1[0]])
                    indi.append(0)
                if (a1[1] >= 0) & (a1[1] <= ((grid_size*grid_size)-1)) & (a not in corn1) & (a1[1] not in c_list):
                    temp_h_val1.append(h_val[a1[1]])
                    indi.append(1)
                if (a1[2] >= 0) & (a1[2] <= ((grid_size*grid_size)-1)) & (a1[2] not in c_list):
                    temp_h_val1.append(h_val[a1[2]])
                    indi.append(2)
                if (a1[3] >= 0) & (a1[3] <= ((grid_size*grid_size)-1)) & (a1[3] not in c_list):
                    temp_h_val1.append(h_val[a1[3]])
                    indi.append(3)
                #print temp_h_val1
                if len(temp_h_val1) > 0:
                    if h_val[current_position_index] <= min(temp_h_val1):
                        p_list[a] = current_position_index
                    else:
                        for i in range(0,len(temp_h_val1),1):
                            if temp_h_val1[i] == min(temp_h_val1):
                                if (p_list[a1[indi[i]]] != (-1)):
                                    p_list[a] = a1[indi[i]]
                                    break
                else:
                    p_list[a] = current_position_index
            del indi[:]
            del a1[:]
            del temp_h_val1[:]
            if (b <= ((grid_size*grid_size)-1)) & (b >= 0) & (k2 == 0):
                o_list.append(b)
                a1.append(b + 1)
                a1.append(b - 1)
                a1.append(b + grid_size)
                a1.append(b - grid_size)
                if (a1[0] >= 0) & (a1[0] <= ((grid_size*grid_size)-1)) & (b not in corn3) & (a1[0] not in c_list):
                    temp_h_val1.append(h_val[a1[0]])
                    indi.append(0)
                if (a1[1] >= 0) & (a1[1] <= ((grid_size*grid_size)-1)) & (b not in corn1) & (a1[1] not in c_list):
                    temp_h_val1.append(h_val[a1[1]])
                    indi.append(1)
                if (a1[2] >= 0) & (a1[2] <= ((grid_size*grid_size)-1)) & (a1[2] not in c_list):
                    temp_h_val1.append(h_val[a1[2]])
                    indi.append(2)
                if (a1[3] >= 0) & (a1[3] <= ((grid_size*grid_size)-1)) & (a1[3] not in c_list):
                    temp_h_val1.append(h_val[a1[3]])
                    indi.append(3)
                #print temp_h_val1,indi
                if len(temp_h_val1) > 0:
                    if h_val[current_position_index] <= min(temp_h_val1):
                        p_list[b] = current_position_index
                    else:
                        for i in range(0, len(temp_h_val1), 1):
                            if temp_h_val1[i] == min(temp_h_val1):
                                if (p_list[a1[indi[i]]] != (-1)):
                                    p_list[b] = a1[indi[i]]
                                    break
                else:
                    p_list[b] = current_position_index
            del indi[:]
            del a1[:]
            del temp_h_val1[:]
            if (c <= ((grid_size*grid_size)-1)) & (c >= 0) & (k3 == 0):
                o_list.append(c)
                a1.append(c + 1)
                a1.append(c - 1)
                a1.append(c + grid_size)
                a1.append(c - grid_size)
                if (a1[0] >= 0) & (a1[0] <= ((grid_size*grid_size)-1)) & (c not in corn3) & (a1[0] not in c_list):
                    temp_h_val1.append(h_val[a1[0]])
                    indi.append(0)
                if (a1[1] >= 0) & (a1[1] <= ((grid_size*grid_size)-1)) & (c not in corn1) & (a1[1] not in c_list):
                    temp_h_val1.append(h_val[a1[1]])
                    indi.append(1)
                if (a1[2] >= 0) & (a1[2] <= ((grid_size*grid_size)-1)) & (a1[2] not in c_list):
                    temp_h_val1.append(h_val[a1[2]])
                    indi.append(2)
                if (a1[3] >= 0) & (a1[3] <= ((grid_size*grid_size)-1)) & (a1[3] not in c_list):
                    temp_h_val1.append(h_val[a1[3]])
                    indi.append(3)
                #print temp_h_val1,indi
                if len(temp_h_val1) > 0:
                    if h_val[current_position_index] <= min(temp_h_val1):
                        p_list[c] = current_position_index
                    else:
                        for i in range(0, len(temp_h_val1), 1):
                            if temp_h_val1[i] == min(temp_h_val1):
                                if (p_list[a1[indi[i]]] != (-1)):
                                    p_list[c] = a1[indi[i]]
                                    break
                else:
                    p_list[c] = current_position_index
            del indi[:]
            del a1[:]
            del temp_h_val1[:]
            if (d <= ((grid_size*grid_size)-1)) & (d >= 0) & (k4 == 0):
                o_list.append(d)
                a1.append(d + 1)
                a1.append(d - 1)
                a1.append(d + grid_size)
                a1.append(d - grid_size)
                if (a1[0] >= 0) & (a1[0] <= ((grid_size*grid_size)-1)) & (d not in corn3) & (a1[0] not in c_list):
                    temp_h_val1.append(h_val[a1[0]])
                    indi.append(0)
                if (a1[1] >= 0) & (a1[1] <= ((grid_size*grid_size)-1)) & (d not in corn1) & (a1[1] not in c_list):
                    temp_h_val1.append(h_val[a1[1]])
                    indi.append(1)
                if (a1[2] >= 0) & (a1[2] <= ((grid_size*grid_size)-1)) & (a1[2] not in c_list):
                    temp_h_val1.append(h_val[a1[2]])
                    indi.append(2)
                if (a1[3] >= 0) & (a1[3] <= ((grid_size*grid_size)-1)) & (a1[3] not in c_list):
                    temp_h_val1.append(h_val[a1[3]])
                    indi.append(3)
                #print temp_h_val1,indi
                if len(temp_h_val1) > 0:
                    if h_val[current_position_index] <= min(temp_h_val1):
                        p_list[d] = current_position_index
                    else:
                        for i in range(0, len(temp_h_val1), 1):
                            if temp_h_val1[i] == min(temp_h_val1):
                                if (p_list[a1[indi[i]]] != (-1)):
                                    p_list[d] = a1[indi[i]]
                                    break
                else:
                    p_list[d] = current_position_index
            del indi[:]
            del a1[:]
            del temp_h_val1[:]
            del temp_f_val[:]
            del temp_g_val[:]
            del index[:]
            for i in range(0, len(o_list), 1):
                temp_f_val.append(f_val[o_list[i]])
                temp_g_val.append(g_val[o_list[i]])
            for i in range(0, len(o_list), 1):
                minimum = min(temp_f_val)
                if temp_f_val[i] != minimum:
                    index.append(i)
            if len(o_list) < 1:
                print "Path not found"
                p = 0
                break
            for i in range(0, len(index), 1):
                temp_g_val[index[i]] = 10000
            new_current_position_index = o_list[temp_g_val.index(min(temp_g_val))]
            for i in range(0, len(o_list), 1):
                if o_list[i] == current_position_index:
                    del o_list[i]
                    break
            current_position_index = new_current_position_index
            current_position = grid[current_position_index]
            #print p_list,current_position_index

    del h_val[:]
    del g_val[:]
    del f_val[:]
    del temp_f_val[:]
    del temp_g_val[:]
    del o_list[:]
    #del c_list[:]
    del index[:]
    #del corn1[:]
    #del corn2[:]
    #del corn3[:]
    #del corn4[:]

    if p != 0:
        i = 1
        path.append(end_point_index)
        a = end_point_index
        while(1):
            if a == start_point_index:
                break
            else:
                path.append(p_list[a])
                a = path[i]
                i = i+1
        path.reverse()

        pix = [[]]
        points = [[]]

        for j in range(((res.shape[0])/(grid_size*2)), (res.shape[0]+1), ((res.shape[0])/(grid_size))):
            for i in range(((res.shape[1])/(grid_size*2)), (res.shape[1]+1) , ((res.shape[1])/(grid_size))):
                pix.append([i, j])
        del pix[0]
        #print pix

        for i in range(0, len(path), 1):
            points.append(pix[path[i]])
        del points[0]
        #print points

        pts = np.array(points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        img = cv2.polylines(res, [pts], False, (0, 255, 255), 3)

        #del pix[:]
        #del pts[:]

        cv2.imshow('img', img)

    for i in range(0,(grid_size*grid_size),1):
            plt.subplot(grid_size,grid_size,i+1), plt.imshow(grid[i]), plt.title(i)
            plt.xticks([]), plt.yticks([])
    plt.show()
    return path

path = astar()
print path


k = 0
y1 = 0

cap = cv2.VideoCapture(0)

while (True):
    ret, frame = cap.read()
    # cv2.imshow('image1',frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #    continue
    gri = []
    gri1 = []
    sd = []

    xstep = frame.shape[0] / grid_size
    ystep = frame.shape[1] / grid_size
    for i in range(0, frame.shape[0], xstep):
        for j in range(0, frame.shape[1], ystep):
            t = frame[i:i + xstep, j:j + ystep]
            t2gray = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
            gri.append(t)
            gri1.append(t2gray)

    frame2hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #lower_red = np.array([150, 80, 0])
    #upper_red = np.array([200, 255, 255])
    lower_red = np.array([0, 100, 0])
    upper_red = np.array([15, 255, 255])
    #lower_green = np.array([0, 100, 0])
    #upper_green = np.array([30, 255, 255])
    lower_green = np.array([10, 100, 100])   #yellow actually
    upper_green = np.array([50, 255, 255])

    masked = cv2.inRange(frame2hsv, lower_red, upper_red)
    masked1 = cv2.inRange(frame2hsv, lower_green, upper_green)

    frame1 = cv2.bitwise_and(frame, frame, mask=masked)
    frame2 = cv2.bitwise_and(frame, frame, mask=masked1)

    frame1 = cv2.medianBlur(frame1, 5)
    frame2 = cv2.medianBlur(frame2, 15)
    masked = cv2.medianBlur(masked, 5)
    masked1 = cv2.medianBlur(masked1, 15)
    frame12gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame22gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    xstep = masked.shape[0] / grid_size
    ystep = masked.shape[1] / grid_size
    for i in range(0, frame.shape[0], xstep):
        for j in range(0, frame.shape[1], ystep):
            t = masked[i:i + xstep, j:j + ystep]
            sd.append(t)

    corners = cv2.goodFeaturesToTrack(frame12gray, 2, 0.1, 40)
    temp1 = []
    if corners is None:
        print "Line not available"
        data.write(struct.pack('!B', int(0)))
    else:
        for m in corners:
            x, y = m.ravel()
            x = int(x)
            y = int(y)
            if (((y - y1)<50)&((y - y1)>-50)):
                temp1.append((x, y))
            y1 = y
            cv2.circle(masked, (x, y), 3, 255, -1)

    corner = cv2.goodFeaturesToTrack(frame22gray, 1, 0.001, 10)
    temp2 = []
    if corner is None:
        print "Dot not available"
        data.write(struct.pack('!B', int(0)))
    else:
        for n in corner:
            x, y = corner.ravel()
            x = int(x)
            y = int(y)
            temp2.append((x, y))
            cv2.circle(masked1, (x, y), 3, 255, -1)
    if (len(temp1) == 1):
        data.write(struct.pack('!B', int(0)))
    if (len(temp1) == 2) & (len(temp2) == 1):
        cv2.line(masked, (temp1[0][0], temp1[0][1]), (temp1[1][0], temp1[1][1]), (0, 255, 0), 3)
        if (temp2[0][1] < temp1[0][1]) & (temp2[0][1] < temp1[1][1]) & (((temp2[0][0] > temp1[0][0]) & (temp2[0][0] < temp1[1][0])) | ((temp2[0][0] < temp1[0][0]) & (temp2[0][0] > temp1[1][0]))):
            #print "North"
            ori = 1
        elif (temp2[0][1] > temp1[0][1]) & (temp2[0][1] > temp1[1][1]) & (((temp2[0][0] > temp1[0][0]) & (temp2[0][0] < temp1[1][0])) | ((temp2[0][0] < temp1[0][0]) & (temp2[0][0] > temp1[1][0]))):
            #print "South"
            ori = 2
        elif (temp2[0][0] < temp1[0][0]) & (temp2[0][0] < temp1[1][0]) & (((temp2[0][1] > temp1[0][1]) & (temp2[0][1] < temp1[1][1])) | ((temp2[0][1] < temp1[0][1]) & (temp2[0][1] > temp1[1][1]))):
            #print "West"
            ori = 3
        elif (temp2[0][0] > temp1[0][0]) & (temp2[0][0] > temp1[1][0]) & (((temp2[0][1] > temp1[0][1]) & (temp2[0][1] < temp1[1][1])) | ((temp2[0][1] < temp1[0][1]) & (temp2[0][1] > temp1[1][1]))):
            #print "East"
            ori = 4
        elif (temp2[0][0] < temp1[0][0]) & (temp2[0][0] < temp1[1][0]) & (temp2[0][1] < temp1[0][1]) & (temp2[0][1] < temp1[1][1]):
            #print "NorthWest"
            ori = 5
        elif (temp2[0][0] > temp1[0][0]) & (temp2[0][0] > temp1[1][0]) & (temp2[0][1] < temp1[0][1]) & (temp2[0][1] < temp1[1][1]):
            #print "NorthEast"
            ori = 6
        elif (temp2[0][0] < temp1[0][0]) & (temp2[0][0] < temp1[1][0]) & (temp2[0][1] > temp1[0][1]) & (temp2[0][1] > temp1[1][1]):
            #print "SouthWest"
            ori = 7
        elif (temp2[0][0] > temp1[0][0]) & (temp2[0][0] > temp1[1][0]) & (temp2[0][1] > temp1[0][1]) & (temp2[0][1] > temp1[1][1]):
            #print "SouthEast"
            ori = 8
        else:
            ori = 0

        if path[k] == path[-1]:
            print "Stop"
            data.write(struct.pack('!B', int(0)))
            break
        #print k
        bot_val = path[k] - path[k + 1]
        if bot_val == 1:
            print "Left"
            if ori == 1:
                print "Turn 90 left"
                data.write(struct.pack('!B', int(3)))
            elif ori == 2:
                print "Turn 90 right"
                data.write(struct.pack('!B', int(2)))
            elif ori == 3:
                print "Forward"
                data.write(struct.pack('!B', int(1)))
            elif ori == 4:
                print "Reverse"
                data.write(struct.pack('!B', int(6)))
            elif ori == 5:
                print "Turn 45 left"
                data.write(struct.pack('!B', int(5)))
            elif ori == 6:
                print "Turn 45 right"
                data.write(struct.pack('!B', int(4)))
            elif ori == 7:
                print "Turn 45 right"
                data.write(struct.pack('!B', int(4)))
            elif ori == 8:
                print "Turn 45 left"
                data.write(struct.pack('!B', int(5)))
            else:
                data.write(struct.pack('!B', int(0)))

        elif bot_val == (-1):
            print "Right"
            if ori == 1:
                print "Turn 90 right"
                data.write(struct.pack('!B', int(2)))
            elif ori == 2:
                print "Turn 90 left"
                data.write(struct.pack('!B', int(3)))
            elif ori == 3:
                print "Reverse"
                data.write(struct.pack('!B', int(6)))
            elif ori == 4:
                print "Forward"
                data.write(struct.pack('!B', int(1)))
            elif ori == 5:
                print "Turn 45 left"
                data.write(struct.pack('!B', int(5)))
            elif ori == 6:
                print "Turn 45 right"
                data.write(struct.pack('!B', int(4)))
            elif ori == 7:
                print "Turn 45 right"
                data.write(struct.pack('!B', int(4)))
            elif ori == 8:
                print "Turn 45 left"
                data.write(struct.pack('!B', int(5)))
            else:
                data.write(struct.pack('!B', int(0)))

        elif bot_val == (-grid_size):
            print "Down"
            if ori == 1:
                print "Reverse"
                data.write(struct.pack('!B', int(6)))
            elif ori == 2:
                print "Forward"
                data.write(struct.pack('!B', int(1)))
            elif ori == 3:
                print "Turn 90 left"
                data.write(struct.pack('!B', int(3)))
            elif ori == 4:
                print "Turn 90 right"
                data.write(struct.pack('!B', int(2)))
            elif ori == 5:
                print "Turn 45 right"
                data.write(struct.pack('!B', int(4)))
            elif ori == 6:
                print "Turn 45 left"
                data.write(struct.pack('!B', int(5)))
            elif ori == 7:
                print "Turn 45 left"
                data.write(struct.pack('!B', int(5)))
            elif ori == 8:
                print "Turn 45 right"
                data.write(struct.pack('!B', int(4)))
            else:
                data.write(struct.pack('!B', int(0)))

        elif bot_val == grid_size:
            print "Up"
            if ori == 1:
                print "Forward"
                data.write(struct.pack('!B', int(1)))
            elif ori == 2:
                print "Reverse"
                data.write(struct.pack('!B', int(6)))
            elif ori == 3:
                print "Turn 90 right"
                data.write(struct.pack('!B', int(2)))
            elif ori == 4:
                print "Turn  90 left"
                data.write(struct.pack('!B', int(3)))
            elif ori == 5:
                print "Turn 45 right"
                data.write(struct.pack('!B', int(4)))
            elif ori == 6:
                print "Turn 45 left"
                data.write(struct.pack('!B', int(5)))
            elif ori == 7:
                print "Turn 45 left"
                data.write(struct.pack('!B', int(5)))
            elif ori == 8:
                print "Turn 45 right"
                data.write(struct.pack('!B', int(4)))
            else:
                data.write(struct.pack('!B', int(0)))

        maxim = []
        for j in range(0, grid_size * grid_size, 1):
            maxim.append(cv2.countNonZero(sd[j]))
        # print m axim
        inde = maxim.index(max(maxim))
        # print inde
        if path[k + 1] == inde:
            k = k + 1
        elif path[k] != inde:
            data.write(struct.pack('!B', int(0)))
            time.sleep(5)
            for j in range(0, grid_size * grid_size, 1):
                maxim.append(cv2.countNonZero(sd[j]))
            inde = maxim.index(max(maxim))
            if path[k] != inde:
                k = 0
                y1 = 0
                ind = inde
                cv2.destroyWindow('img')
                del path[:]
                path = astar()
                print path

    cv2.imshow('image4', masked)
    cv2.imshow('image5', masked1)
    l = cv2.waitKey(3) & 0xff
    if l == 27:
        break

cv2.destroyAllWindows()
cap.release()