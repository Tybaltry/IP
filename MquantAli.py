import numpy as np 
import cv2
import matplotlib.pyplot as plt 
from sklearn import linear_model
from scipy.optimize import curve_fit
from pythreshold.utils import kapur_threshold, test_thresholds

def apply_threshold(img, threshold=128, wp_val=255):
     return ((img >= threshold) * wp_val).astype(np.uint8)
 
###Reading and transforming of standard images
avrages = []
for i in [10, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 400, 500]:
    calibrated = []
    temp = [0,0,0,0,0]

    filename = f'calibrationMquant/C{i}-1.jpg'
    #print(filename)
    
    #Gathering and splitting of all image channels
    gray_image = cv2.imread(filename,0)
    c_images = cv2.imread(filename,1)
    r,g,b = cv2.split(c_images)
    h_image = cv2.cvtColor(c_images, cv2.COLOR_BGR2HSV) 
    h,s,v = cv2.split(h_image)
    
    #Threshold and binary transformation
    img = gray_image
    th = kapur_threshold(img)
    #plt.imshow(apply_threshold(img, th), cmap='gray')
    thresh = apply_threshold(img, th)
    binary = np.where(thresh < 0, thresh, thresh + 1)
 
    #Removal of thresholded values from each channel
    gray_images = np.multiply(binary, gray_image)
    h_images = np.multiply(binary,h)
    r_images = np.multiply(binary,r)
    g_images = np.multiply(binary,g)
    b_images = np.multiply(binary,b)
    
    #Averaging of each channel, excluding all zero values
    gray_intensities = np.average(gray_images, axis=(0,1), weights=(gray_images>0.5))
    h_intensities = np.average(h_images, axis=(0,1), weights=(h_images>0.5))
    r_intensities = np.average(r_images, axis=(0,1), weights=(r_images>0.5))
    g_intensities = np.average(g_images, axis=(0,1), weights=(g_images>0.5))
    b_intensities = np.average(b_images, axis=(0,1), weights=(b_images>0.5))
        
    #Combining and averaging of replicates for each channel
    calibrated += [gray_intensities, h_intensities, r_intensities, g_intensities, b_intensities]

    for k in range(5):
        temp[k] = np.average(calibrated[k::5])
    avrages.append(temp)
#print(avrages)

#Creating the averages into a callable array, gray = 0, hue = 1, red = 2, green = 3, blue = 4
channels = [[],[],[],[],[]]
for i in avrages:
    channels[0].append(i[0])
    channels[1].append(i[1])
    channels[2].append(i[2])
    channels[3].append(i[3])
    channels[4].append(i[4])
channelsarray = np.asarray(channels)        
#print(channels[0])

##Shaping and calling of x and y varaibles for all channels
xdf = np.array([10, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 400, 500])
xdfa = xdf.reshape(13, 1)


gray_ydf = channels[0]
h_ydf = channels[1]
r_ydf = channels[2]
g_ydf = channels[3]
b_ydf = channels[4]


##Exponential Regression
#Change ydf to match wanted/respective channel
#Run section for wanted channel, then predict below   
def GrayExp(gray_x, gray_a, gray_b, gray_c):
    return gray_a * np.exp(-gray_b * gray_x) + gray_c
p0 = (200, .1, 50)
params, cv = curve_fit(GrayExp, xdf, gray_ydf, p0)
gray_a, gray_b, gray_c = params

#R2 Value
squaredDiffs = np.square(gray_ydf - GrayExp(xdf, gray_a, gray_b, gray_c))
squaredDiffsFromMean = np.square(gray_ydf - np.mean(gray_ydf))
rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
print(f"Gray R² = {rSquared}")

#Exponential Curve Plot 
plt.plot(xdf, gray_ydf, '.', label="data")
plt.plot(xdf, GrayExp(xdf, gray_a, gray_b, gray_c), '--', label="fitted", color="gray")
plt.title("Gray Channel Exponential Curve")
plt.xlabel('Concentration')
plt.ylabel('Intensity')

#Inspect the parameters and return equation
print(f"(Gray) Y = {gray_a} * e^(-{gray_b} * gray_x) + {gray_c}")


##HUE 
def HueExp(h_x, h_a, h_b, h_c):
    return h_a * np.exp(-h_b * h_x) + h_c
p0 = (200, .1, 50)
params, cv = curve_fit(HueExp, xdf, h_ydf, p0)
h_a, h_b, h_c = params

#R2 Value
squaredDiffs = np.square(h_ydf - HueExp(xdf, h_a, h_b, h_c))
squaredDiffsFromMean = np.square(h_ydf - np.mean(h_ydf))
rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
print(f"Hue R² = {rSquared}")

#Exponential Curve Plot 
plt.plot(xdf, h_ydf, '.', label="data")
plt.plot(xdf, HueExp(xdf, h_a, h_b, h_c), '--', label="fitted", color="yellow")
plt.title("Gray Channel Exponential Curve")
plt.xlabel('Concentration')
plt.ylabel('Intensity')

#Inspect the parameters and return equation
print(f"(Hue) Y = {h_a} * e^(-{h_b} * h_x) + {h_c}")


##RED
def RedExp(r_x, r_a, r_b, r_c):
    return r_a * np.exp(-r_b * r_x) + r_c
p0 = (50, .1, 50)
params, cv = curve_fit(RedExp, xdf, r_ydf, p0)
r_a, r_b, r_c = params

#R2 Value
squaredDiffs = np.square(r_ydf - RedExp(xdf, r_a, r_b, r_c))
squaredDiffsFromMean = np.square(r_ydf - np.mean(r_ydf))
rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
print(f"Red R² = {rSquared}")

#Exponential Curve Plot 
plt.plot(xdf, r_ydf, '.', label="data")
plt.plot(xdf, RedExp(xdf, r_a, r_b, r_c), '--', label="fitted", color="red")
plt.title("Gray Channel Exponential Curve")
plt.xlabel('Concentration')
plt.ylabel('Intensity')

#Inspect the parameters and return equation
print(f"(Red) Y = {r_a} * e^(-{r_b} * x) + {r_c}")


##GREEN
def GreenExp(g_x, g_a, g_b, g_c):
    return g_a * np.exp(-g_b * g_x) + g_c
p0 = (200, .1, 50)
params, cv = curve_fit(GreenExp, xdf, g_ydf, p0)
g_a, g_b, g_c = params

#R2 Value
squaredDiffs = np.square(g_ydf - GreenExp(xdf, g_a, g_b, g_c))
squaredDiffsFromMean = np.square(g_ydf - np.mean(g_ydf))
rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
print(f"Green R² = {rSquared}")

#Exponential Curve Plot 
plt.plot(xdf, g_ydf, '.', label="data")
plt.plot(xdf, GreenExp(xdf, g_a, g_b, g_c), '--', label="fitted", color="green")
plt.title("All Channels Exponential Curve")
plt.xlabel('Concentration')
plt.ylabel('Intensity')

#Inspect the parameters and return equation
print(f"(Green) Y = {g_a} * e^(-{g_b} * x) + {g_c}")


##BLUE
def BlueExp(b_x, b_a, b_b, b_c):
    return b_a * np.exp(-b_b * b_x) + b_c
p0 = (200, .1, 50)
params, cv = curve_fit(BlueExp, xdf, b_ydf, p0, maxfev = 1500)
b_a, b_b, b_c = params

#R2 Value
squaredDiffs = np.square(b_ydf - BlueExp(xdf, b_a, b_b, b_c))
squaredDiffsFromMean = np.square(b_ydf - np.mean(b_ydf))
rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
print(f"Blue R² = {rSquared}")

#Exponential Curve Plot 
plt.plot(xdf, b_ydf, '.', label="data")
plt.plot(xdf, BlueExp(xdf, b_a, b_b, b_c), '--', label="fitted", color="blue")
plt.title("All Channels Exponential Curves")
plt.xlabel('Concentration')
plt.ylabel('Intensity')

#Inspect the parameters and return equation
print(f"(Blue) Y = {b_a} * e^(-{b_b} * b_x) + {b_c}")


###Reading of unknown image(s) and prediction
def uapply_threshold(uimg, uthreshold=128, uwp_val=255):
    return ((uimg >= uthreshold) * uwp_val).astype(np.uint8)

for h in [6.5, 15, 25, 35, 45, 65, 80, 175, 250, 350, 450]:
    unkfile = f'unknownsMquant/Ex{h}-1.jpg'
    gray_unknown = cv2.imread(unkfile,0)
    c_unknowns = cv2.imread(unkfile,1)
    r_unknown,g_unknown,b_unknown = cv2.split(c_unknowns)
    h_unknowns = cv2.cvtColor(c_unknowns, cv2.COLOR_BGR2HSV) 
    h_unknown,s_unknown,v_unknown = cv2.split(h_unknowns)
    
    #Thresholding and binary conversion of all channels
    uimg = gray_unknown
    uth = kapur_threshold(uimg)
    #plt.imshow(uapply_threshold(uimg, uth), cmap='gray')
    uthresh = uapply_threshold(uimg, uth)
    binary_unknowns = np.where(uthresh < 0, uthresh, uthresh + 1)
    
    
    grayt_unknown = np.multiply(binary_unknowns, gray_unknown)
    ht_unknown = np.multiply(binary_unknowns, h_unknown)
    rt_unknown = np.multiply(binary_unknowns, r_unknown)
    gt_unknown = np.multiply(binary_unknowns, g_unknown)
    bt_unknown = np.multiply(binary_unknowns, b_unknown)
    
    
    #Averaging of all channels
    grayt_intensities = np.average(grayt_unknown, axis=(0,1), weights=(grayt_unknown>0.5))
    ht_intensities = np.average(ht_unknown, axis=(0,1), weights=(ht_unknown>0.5))
    rt_intensities = np.average(rt_unknown, axis=(0,1), weights=(rt_unknown>0.5))
    gt_intensities = np.average(gt_unknown, axis=(0,1), weights=(gt_unknown>0.5))
    bt_intensities = np.average(bt_unknown, axis=(0,1), weights=(bt_unknown>0.5))
    print(unkfile)
    
    #Exponential Regression Prediction -- Use Equation!
    #GRAY
    grayexp_predict = (np.log((grayt_intensities - gray_c)/gray_a))/-gray_b
    print("Gray =", grayexp_predict)
    GrayGT=np.interp(grayt_intensities, gray_ydf[::-1], xdf[::-1])
    #print("Gray Interp = ", GrayGT)
    
    
    #HUE
    hexp_predict = (np.log(abs(ht_intensities - h_c)/h_a))/-h_b
    print("Hue =", hexp_predict)
    HGT=np.interp(ht_intensities, h_ydf[::-1], xdf[::-1])
    #print("Hue Interp = ", HGT)
    
    #RED
    rexp_predict = (np.log(abs(rt_intensities - r_c)/r_a))/-r_b
    print("Red =", rexp_predict)
    RGT=np.interp(rt_intensities, r_ydf[::-1], xdf[::-1])
    #print("Red Interp = ", RGT)
    
    
    #GREEN
    gexp_predict = (np.log(abs(gt_intensities - g_c)/g_a))/-g_b
    print("Green =", gexp_predict)
    GGT=np.interp(gt_intensities, g_ydf[::-1], xdf[::-1])
    #print("Green Interp = ", GGT)
    
    
    #BLUE
    bexp_predict = (np.log(abs(bt_intensities - b_c)/b_a))/-b_b
    print("Blue = ", bexp_predict)
    BGT=np.interp(bt_intensities, b_ydf[::-1], xdf[::-1])
    #print("Blue Interp = ", BGT)
'''