
class ImageRecord:     
    
    def __init__(     
        self,          
        image,          
        label,          
        width,          
        height,         
        mode            # 图像模式（如RGB，L，灰度等）
    ):
        self.image = image
        self.label = label
        self.width = width
        self.height = height
        self.mode = mode   # 将图像数据、标签、宽度、高度和模式赋值给对象属性
        
        # 这就是我的统一数据对象，包含了图像数据和相关信息，可以在后续的处理和分析中使用