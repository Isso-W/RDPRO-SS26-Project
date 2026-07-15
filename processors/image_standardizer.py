import os

from PIL import Image


class ImageStandardizer:
    # Standardize images before model training

    def __init__(
        self,
        output_dir: str,
        target_size: tuple[int, int] = (224, 224),
        resize_mode: str = "letterbox_resize"
    ):
        # Initialize processing configuration

        self._output_dir = output_dir

        self._target_size = target_size

        self._resize_mode = resize_mode

        # Cache created directories
        self._created_dirs = set()

        os.makedirs(
            output_dir,
            exist_ok=True
        )

    def process(
        self,
        dataset
    ) -> dict:
        # Main entry point

        return self._standardize_images(
            dataset
        )

    def _standardize_images(
        self,
        dataset
    ) -> dict:
        # Convert, resize and save all images

        processed_images = 0

        converted_to_rgb = 0

        resized_images = 0

        already_target_size = 0

        format_converted = 0

        # Track image index for each class
        label_indices = {}

        for split in dataset.keys():

            label_indices[split] = {}

            # Get class label names
            label_feature = (
                dataset[split]
                .features["label"]
            )

            for sample in dataset[split]:

                image = sample["image"]

                # Convert label id to class name
                label = label_feature.int2str(
                    sample["label"]
                )

                # Convert non-RGB images to RGB
                if image.mode != "RGB":

                    image = self._convert_to_rgb(
                        image
                    )

                    converted_to_rgb += 1

                # Resize images to target size
                if (
                    image.width,
                    image.height
                ) != self._target_size:

                    image = self._resize_image(
                        image
                    )

                    resized_images += 1
                else:
                    already_target_size += 1

                index = label_indices[
                    split
                ].get(
                    label,
                    0
                )

                # Check if original format is not JPEG
                original_format = image.format
                if original_format != "JPEG":

                    format_converted += 1

                # Save processed image
                self._save_image(
                    image=image,
                    split=split,
                    label=label,
                    index=index
                )

                label_indices[
                    split
                ][label] = index + 1

                processed_images += 1

                if processed_images % 1000 == 0:

                    print(
                        f"Processed {processed_images} images..."
                    )

        return {

            "processed_images":
                processed_images,

            "converted_to_rgb":
                converted_to_rgb,

            "resized_images":
                resized_images,

            "already_target_size":
                already_target_size,

            "resize_mode":
                self._resize_mode,

            "target_size":
                self._target_size,

            "output_directory":
                self._output_dir,

            "format_converted":
                format_converted,

            "output_format":
                "JPEG"
        }

    def _convert_to_rgb(
        self,
        image: Image.Image
    ) -> Image.Image:
        # Convert image to RGB format

        return image.convert(
            "RGB"
        )

    def _resize_image(
        self,
        image: Image.Image
    ) -> Image.Image:
        # Select resize strategy

        if (
            self._resize_mode
            ==
            "letterbox_resize"
        ):

            return self._letterbox_resize(
                image
            )

        return self._direct_resize(
            image
        )

    def _direct_resize(
        self,
        image: Image.Image
    ) -> Image.Image:
        # Resize image directly

        return image.resize(
            self._target_size,
            Image.LANCZOS
        )

    def _letterbox_resize(
        self,
        image: Image.Image
    ) -> Image.Image:
        # Resize while preserving aspect ratio

        target_w, target_h = (
            self._target_size
        )

        # Calculate scaling factor
        scale = min(
            target_w / image.width,
            target_h / image.height
        )

        new_w = int(
            image.width * scale
        )

        new_h = int(
            image.height * scale
        )

        resized = image.resize(
            (new_w, new_h),
            Image.LANCZOS
        )

        # Create padded canvas
        canvas = Image.new(
            "RGB",
            self._target_size,
            (0, 0, 0)
        )

        offset_x = (
            target_w - new_w
        ) // 2

        offset_y = (
            target_h - new_h
        ) // 2

        # Center image on canvas
        canvas.paste(
            resized,
            (offset_x, offset_y)
        )

        return canvas

    def _save_image(
        self,
        image: Image.Image,
        split: str,
        label: str,
        index: int
    ) -> None:
        # Save image to output directory

        split_label_dir = os.path.join(
            self._output_dir,
            split,
            label
        )

        # Create class directory if needed
        if (
            split_label_dir
            not in self._created_dirs
        ):

            os.makedirs(
                split_label_dir,
                exist_ok=True
            )

            self._created_dirs.add(
                split_label_dir
            )

        # Generate filename
        filename = f"{index:06d}.jpg"

        filepath = os.path.join(
            split_label_dir,
            filename
        )

        # Save as high-quality JPEG
        image.save(
            filepath,
            format="JPEG",
            quality=95,
            optimize=True
        )
