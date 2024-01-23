import cv2
import matplotlib.pyplot as plt

def extract_corners(image_path, threshold_value=10, epsilon_factor=0.04):
    # 讀取圖像
    image = cv2.imread(image_path)

    # 將圖像轉換為灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 進行二值化處理
    _, binary_image = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    # 找到輪廓和層次結構
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # 提取內層輪廓的索引
    inner_contour_indices = [i for i, h in enumerate(hierarchy[0]) if h[3] != -1]

    # 提取內層輪廓
    inner_contours = [contours[i] for i in inner_contour_indices]
    
    largest_contour = max(inner_contours, key=cv2.contourArea)

    # 提取角點
    corners = []
    epsilon = epsilon_factor * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    corners.extend(approx[:, 0, :])

    return corners

# 調用函數取得角點座標
corners = extract_corners('data/6.jpg')
#print(corners)

# 顯示結果
image = cv2.imread('data/6.jpg')
for corner in corners:
    x, y = corner
    cv2.circle(image, (x, y), 15, (0, 255, 0), -1)
    print(f"({x}, {y})")

plt.subplot(111), plt.axis('off'), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Image with Corners')
plt.show()