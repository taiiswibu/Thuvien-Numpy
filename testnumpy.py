import numpy as np 
from numpy import pi
import cv2
import pandas as pd
import sys


def tinhtoan():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    
    # Phép cộng hai mảng
    c = a + b
    
    # Phép nhân hai mảng
    d = a * b
    
    # Tính tổng các phần tử trong mảng
    e = np.sum(a)
    
    # Tìm giá trị lớn nhất trong mảng
    f = np.max(a)
    
    # Tìm giá trị nhỏ nhất trong mảng
    g = np.min(a)
    print("công hai mảng là:",c)
    print("nhân hai mang là :",d)
    print("tổng các phần tử:",e)
    print("giá trị lớn nhất của mảng",f)
    print("giá trị nhỏ nhất của mảng",g)
    

def taomang():

    # Tạo một mảng 1 chiều có 5 phần tử với giá trị ban đầu là 0
    a = np.zeros(5)
    
    # Tạo một mảng 2 chiều kích thước 3x3 với giá trị ban đầu là 1
    b = np.ones((3, 3))
    
    # Tạo một mảng 1 chiều có 5 phần tử với giá trị ngẫu nhiên từ 0 đến 1
    c = np.random.rand(5)
    
    print("tao mảng có phần tử 0-5",a)
    print("tạo mảng hai chiều kích thước 3x3",b)
    print("tạo mảng một chiều random có 5pt",c)
    

def truycap():

    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    
    # Lấy phần tử ở hàng thứ 2, cột thứ 3
    b = a[1, 2]
    
    # Lấy hàng thứ 2
    c = a[1, :]
    
    # Lấy cột thứ 3
    d = a[:, 2]
    
    # Thay đổi giá trị của phần tử ở hàng thứ 1, cột thứ 2
    a[0, 1] = 10
    
    print(" Lấy phần tử ở hàng thứ 2, cột thứ 3: ",b)
    print(" Lấy phần tử ở hàng thứ 2:",c)
    print(" Lấy phần tử ở cột thứ 3: ",d)
    print(" thay đổi giá trị :",a)
    print("độ lớn cảu mảng",a.size)
    
    
    
def dstt():

    # Tạo ma trận ngẫu nhiên kích thước 3x3 với giá trị từ 0 đến 1
    a = np.random.rand(2,2)
    
    # Tính định thức của ma trận
    b = np.linalg.det(a)
    
    # Tính ma trận nghịch đảo
    c = np.linalg.inv(a)
    
    # Tính giá trị riêng và vector riêng của ma trận
    d, e = np.linalg.eig(a)

    print("ma trận ngẫu nhiên đc tạo là :",a)
    print("định thức ma trận:",b)
    print("ma trận nghịch đảo:",c)
    print("tính giá trị riêng vecto riêng:",d,e)
    

def xulyhinhanh():
    
    # Đọc ảnh từ file
    img = cv2.imread('anh.png')
    
    # Chuyển đổi ảnh thành ma trận NumPy
    img_array = np.array(img)
    
    # Cắt ảnh thành kích thước mới
    cropped_img = img_array[100:300, 200:400]
    
    # Thay đổi kích thước ảnh
    resized_img = cv2.resize(img_array, (500, 500))
    
    # Chuyển đổi ảnh sang ảnh xám
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Tìm cạnh của ảnh
    edges = cv2.Canny(img, 100, 200)
    
    print(img_array)
    
def luonggiac():
    
    # Tạo một mảng numpy với các giá trị từ 0 đến 3.14
    x = np.linspace(0, np.pi, 10)
    
    # Tính sin(x), cos(x), tan(x) của mảng x
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    tan_x = np.tan(x)
    
    print(sin_x)
    print(cos_x)
    print(tan_x)

def loga():
   
    # Tạo một mảng numpy với các giá trị từ 1 đến 10
    x = np.linspace(1, 10, 10)
    
    # Tính logarithm và exponential của mảng x
    log_x = np.log(x)
    exp_x = np.exp(x)
    
    print(log_x)
    print(exp_x)
    
def maxmin():
   
    # Tạo hai mảng numpy với các giá trị ngẫu nhiên
    a = np.random.rand(5)
    b = np.random.rand(5)
    
    # Tìm giá trị lớn nhất và nhỏ nhất của mảng a và b
    max_a = np.max(a)
    max_b = np.max(b)
    min_a = np.min(a)
    min_b = np.min(b)
    
    # Tìm giá trị lớn nhất và nhỏ nhất của hai mảng a và b
    max_ab = np.maximum(a, b)
    min_ab = np.minimum(a, b)

    print(max_a)
    print(min_ab)

def sum_prod():

    # Tạo một mảng numpy với các giá trị ngẫu nhiên
    a = np.random.rand(3, 3)
    
    # Tính tổng và tích các phần tử của mảng a
    sum_a = np.sum(a)
    prod_a = np.prod(a)
    
    # Tính tổng và tích các phần tử của mảng a theo từng trục
    sum_a_axis0 = np.sum(a, axis=0)
    sum_a_axis1 = np.sum(a, axis=1)
    prod_a_axis0 = np.prod(a, axis=0)
    prod_a_axis1 = np.prod(a, axis=1)
    print(sum_a)
    print(prod_a)
    print(sum_a_axis0)
    print(prod_a_axis0)


def testnumpy():
    
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = a.size
    c = a.shape
    d = a.ndim
    c1 = np.array([[1, 2], [3, 4]], dtype=complex)
    e = np.empty((2, 3))
    a1 =np.arange(10, 30, 5)
    np.linspace(0, 2, 9) 
    x = np.linspace(0, 2 * pi, 100)
    f = np.sin(x)
    #print(np.set_printoptions(threshold=sys.maxsize))
    b1 = np.arange(12).reshape(4, 3)
    rg = np.random.default_rng(100)
    print(rg.random((2, 3)))
    
  
def docdata():
    # biến a sẽ bị lỗi và xuất ra giá trị 
    a = np.genfromtxt('test1.txt', delimiter=' ')
    # Để khắc phục điều đó ta sử dụng như biến 
    b= np.genfromtxt('test1.txt', delimiter=',', names=True, dtype=None, encoding=None)
    print(b)
    #print(b[b>10])
    b1 = np.array([1,2,3])
    print(b1 [b1 >1])
    
    
    
docdata()






