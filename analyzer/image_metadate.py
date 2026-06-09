
class ImageMetadataExtractor:   
    
    def extract(self, image):    
        
        return {
            "width": image.width,    
            "height": image.height,
            "mode": image.mode
        }