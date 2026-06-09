from collections import Counter
from posixpath import split


class ImageStatisticsAnalyzer:
    
    def analyze(self, dataset):
        
        report = {}
        
        report.update(
            self._dataset_statistics(dataset)
        )
        
        report.update(
            self._metadata_statistics(dataset)
        )
        
        report.update(
            self._mode_distribution(dataset)
        )
        
        report.update(
            self._format_distribution(dataset)
        )
       
        return report
    
    def _dataset_statistics(self, dataset):
        
        split_sizes = {

            split: len(dataset[split])

            for split in dataset.keys()
        }
        
        total_images = sum(split_sizes.values())
        
        labels = [
            sample["label"] for sample in dataset["train"]
        ]
        
        num_classes = len(set(labels))
        
        class_distribution = dict(Counter(labels))
        
        return {
            "split_sizes": split_sizes,
            "total_images": total_images,
            "num_classes": num_classes,
            "class_distribution": class_distribution
        }
        
    def _metadata_statistics(self, dataset):
            
        widths = []
        heights = []
            
        for split in dataset.keys():

            for sample in dataset[split]:
                
                image = sample["image"]
                    
                widths.append(image.width)
                    
                heights.append(image.height)
                
                
        return {
            "min_width": min(widths),
            "max_width": max(widths),
            "avg_width": sum(widths)/len(widths),
            "min_height": min(heights),
            "max_height": max(heights),
            "avg_height": sum(heights)/len(heights)  
        }
            
    def _mode_distribution(self, dataset):

        modes = []
        
        for split in dataset.keys():

            for sample in dataset[split]:
                
                modes.append(sample["image"].mode)
            
        return {
            "mode_distribution": dict(Counter(modes))
        }
        
    def _format_distribution(self, dataset):
        
        formats = []
        
        for split in dataset.keys():

            for sample in dataset[split]:
            
                image = sample["image"]
                
                formats.append(str(image.format))
            
        return {
            
            "format_distribution": dict(Counter(formats))
        }