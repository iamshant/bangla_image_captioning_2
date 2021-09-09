from utils_1 import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    # create_input_files(dataset='flickr8k',
    #                    karpathy_json_path='/content/gdrive/MyDrive/image_captioning/a-PyTorch-Tutorial-to-Image-Captioning/caption_datasets/dataset_flickr8k.json',
    #                    image_folder='/content/gdrive/MyDrive/image_captioning/flickr8k/images/',
    #                    captions_per_image=5,
    #                    min_word_freq=5,
    #                    output_folder='/content/gdrive/MyDrive/image_captioning/a-PyTorch-Tutorial-to-Image-Captioning/caption data',
    #                    max_len=50)
    create_input_files(dataset='flickr8k',
                       karpathy_json_path='/content/gdrive/MyDrive/image_captioning/bangla_dataset/train_test_val_2.json',
                    #    image_folder='/content/gdrive/MyDrive/image_captioning/bangla_dataset/final_image_dataset_7468/',
                    image_folder='/content/gdrive/MyDrive/image_captioning/bangla_dataset/final_image_dataset_7468_with_all_RGB',
                       captions_per_image=3,
                       min_word_freq=3,
                       output_folder='/content/gdrive/MyDrive/image_captioning/a-PyTorch-Tutorial-to-Image-Captioning/caption_data_2',
                       max_len=30)