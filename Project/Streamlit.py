import streamlit as st 
import pandas as pd 
import numpy as np 
from PIL import Image, ImageDraw
from utils import load_session
from main import *
import os

session = load_session(PATH_MODEL)

class CFG:
    image_size = IMAGE_SIZE
    conf_thres = 0.05
    iou_thres = 0.3

cfg = CFG()

### Header
st.title('Object Detection Web')
st.subheader('AIO2024 - Đặng Phúc Bảo Châu')

#### Chọn và hiển thị hình ảnh ###

# Nút upload
uploaded_file = st.file_uploader('Choose an image to detect: ', type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Đọc, lưu và hiển thị ảnh
    with open(os.path.join("uploaded_images", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getvalue())
        
    image = Image.open(uploaded_file)
    st.image(image, caption='Your image', use_column_width=True)

### Nhấn process và tiến hành đưa hình ảnh vào model xử lý ###
    
    # Nút Process
    if st.button('Process'):
        # Lấy danh sách tất cả các tệp trong thư mục uploaded_images
        image_files = os.listdir('uploaded_images')

        # Kiểm tra xem có ảnh nào trong thư mục hay không
        if image_files:
            # Lấy tên của ảnh đầu tiên trong thư mục
            first_image_name = image_files[0]
            
            # Đường dẫn đầy đủ của ảnh
            image_path = os.path.join('uploaded_images' , first_image_name)
            
            # Mở ảnh và gán vào biến image
            image = cv2.imread(image_path)
            pred = prediction(
                        session=session,
                        image=image,
                        cfg=cfg
                        )

### Hiển thị hình ảnh sau khi xử lí ###
            
            # Chuyển đổi image và pred thành đối tượng PIL Image và bounding boxes
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            bounding_boxes = np.array(pred)[:, :4]  # Lấy tọa độ của bounding boxes

            # Visualize hình ảnh với bounding box
            visualized_image = visualize(image_pil, bounding_boxes)

            # Hiển thị hình ảnh đã visualize
            st.image(visualized_image, caption='Image after processing', use_column_width=True)

            # Lấy danh sách tất cả các tệp trong thư mục
            file_list = os.listdir('uploaded_images')

            # Duyệt qua từng tệp và xóa nó
            for file_name in file_list:
                file_path = os.path.join('uploaded_images', file_name)
                os.remove(file_path)
else:
    st.write('No image founded !')


