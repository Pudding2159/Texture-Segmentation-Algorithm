from Segmentation_algorithm import Segmentation_ALG1
from Process_Image_ import slic_zero_segmentation

def main():
    IMAGE_FILE = "./texture_t/tm3_1_1.png"
    image_slic = slic_zero_segmentation(IMAGE_FILE,500,0.5)
    Segmentation_ALG1('./Prepared_Image.png',20)
    # Segmentation_ALG1(IMAGE_FILE)

if __name__ == "__main__":
    main()