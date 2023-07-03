from PIL import Image

window_size = 128
step_size = 120
# 给图片padding (window_size-step_size)/2个像素
def expand_image(image_path):
    # 读取大图片
    original_image = Image.open(image_path)
    # 获取大图片的宽度和高度
    width, height = original_image.size
    # 创建新的空白图片
    minus1 = window_size - step_size
    expanded_image = Image.new("RGB", (width + minus1, height + minus1))
    # 将大图片绘制到新图片上
    expanded_image.paste(original_image, (int(minus1/2), int(minus1/2)))
    # 返回新的图片
    return expanded_image


image_path = "F:/00UNET/CroppingCraft/2-6.jpg"
expanded_image = expand_image(image_path)

def save_sliding_window_images(image_path, window_size, step_size, output_path):
    # 读取大图片
    original_image = Image.open(image_path)
    # 获取大图片的宽度和高度
    width, height = original_image.size
    # 定义窗口的左上角坐标
    x = 0
    y = 0

    # 滑动窗口并保存图片
    while y + window_size <= height:
        while x + window_size <= width:
            # 截取窗口内的图片
            window_image = original_image.crop((x, y, x + window_size, y + window_size))
            # 保存窗口内的图片
            window_image.save(output_path + f"{int(x/step_size)}_{int(y/step_size)}.jpg")
            # 向右滑动
            x += step_size

        # 滑动到最右端后回到本行的起点
        x = 0

        # 向下滑动
        y += step_size

# 使用示例

output_path = "F:/00UNET/CroppingCraft/result/"

save_sliding_window_images(image_path, window_size, step_size, output_path)


'''
from PIL import Image
window_size = 128
step_size = 128


image_path = "F:/00UNET/CroppingCraft/2-6.jpg"

def save_cropping_images(image_path, window_size, step_size, output_path):
    # 读取大图片
    original_image = Image.open(image_path)
    # 获取大图片的宽度和高度
    width, height = original_image.size
    # 定义窗口的左上角坐标
    x = 0
    y = 0

    # 滑动窗口并保存图片
    while y + window_size <= height:
        while x + window_size <= width:
            # 截取窗口内的图片
            window_image = original_image.crop((x, y, x + window_size, y + window_size))
            # 保存窗口内的图片
            window_image.save(output_path + f"{int(x/step_size)}_{int(y/step_size)}.jpg")
            # 向右滑动
            x += step_size

        # 滑动到最右端后回到本行的起点
        x = 0

        # 向下滑动
        y += step_size

# 使用示例
output_path = "F:/00UNET/CroppingCraft/result-nowindows/"

save_cropping_images(image_path, window_size, step_size, output_path)



'''