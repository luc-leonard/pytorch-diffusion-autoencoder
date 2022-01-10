import glob
import os
from functools import partial
from pathlib import Path

from tqdm import tqdm


def resize_image(filename, size, base_folder, output_folder):
    from PIL import Image

    with Image.open(filename) as img:
        img = img.resize(size, Image.ANTIALIAS)
        output_path = Path(output_folder) / Path(filename).relative_to(base_folder)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # print(f'{filename} => {output_path}')
        img.save(str(output_path))


def batch_resize_images(image_folder, output_folder, size):
    """
    Resize all images in a folder to a specified size.
    """
    import os

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    from multiprocessing import Pool

    pool = Pool(processes=4)
    input_files = glob.glob(image_folder + "/**/*.png")
    it = pool.imap(
        partial(
            resize_image,
            base_folder=image_folder,
            size=size,
            output_folder=output_folder,
        ),
        input_files,
    )
    resuts = list(tqdm(it, total=len(input_files)))
    pool.close()
    pool.join()


batch_resize_images(
    "/media/lleonard/big_slow_disk/datasets/ffhq/images1024x1024/",
    "./data/images_resized",
    (64, 64),
)
