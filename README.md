# Font Classification

Official PyTorch implementation of "Investigating the Effect of using Synthetic and Semi-synthetic Images for Historical Document Font Classification"

Font classification using pre-trained CNNs.

The classification is made on extracted patches. The pre-processing steps are: removing black borders around the document pages, cropping the image around the text and then creating the patches. 

Semi-synthetic generated images using DocCreator and Synthetic images using OpenGAN.

The generated data (patches, synthetic, and semi-synthetic) can be found in Zenodo: (link to be shared soon)

The dataset class takes images and labels from .csv file in the following format [image_name, label]

### Models

* VGG 19 with Batch Norm: 'vgg' 
* ResNet-18: 'resnet18' 
* ResNet-50: 'resnet50'
* DenseNet-201: 'densenet'
* EfficientNet-b0: 'efficientnet'

### Training with 10K baseline patches

```bash
python classifier_font.py --batch_size 32 --num_classes 10 --train_dir /path/to/patches/train/images/ --train_csv /path/to/patches/training.csv --val_dir /path/to/patches/validation/images/ --val_csv /path/to/patches/validation.csv
```

### Training with 10K baseline patches + DocCreator patches

```bash
python classifier_font_combined.py --batch_size 32 --num_classes 10 --train_dir /path/to/patches/train/images/ --train_dir_gan /path/to/doccreator_patches/ --train_csv /path/to/patches/training.csv --train_csv_gan /path/to/doccreator_labels.csv --val_dir /path/to/patches/validation/images/ --val_csv /path/to/patches/validation.csv
```

### Test Need to fix and upload test code

```bash
python test_classifier.py --model resnet50 --path_to_model /path/to/trained/model --batch_size 64 --num_classes 10 --image_dir /path/to/images/ --csv_file /path/to/csv/file.csv
```

### Citation

If you find this useful in your research, consider citing our work published in the Document Analysis Systems, DAS 2022:

```
@inproceedings{Nikolaidou2022InvestigatingTE,
  title={Investigating the Effect of Using Synthetic and Semi-synthetic Images for Historical Document Font Classification},
  author={Konstantina Nikolaidou and Richa Upadhyay and Mathias Seuret and Marcus Liwicki},
  booktitle={DAS},
  year={2022}
}
```
