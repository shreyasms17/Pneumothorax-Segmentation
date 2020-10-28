#Pneumothorax-Segmentation#

###ABSTRACT###
Pneumothorax is a medical condition in which there is an air leak between the chest wall and lungs. It is diagnosed by radiologists and it’s confirmation is a tedious task. Semantic image segmentation carried out using the Attention UNet and the Attention UNet + ResNet34 models, is used to detect pneumothorax in the radiological images, hence providing non-radiologists with confident results. Additionally, the difference in performance with conventional ways of training the models is analyzed and compared with contemporary methodologies used like snapshot callbacks and stochastic weighting average. The motivation for our approach is that the incorporation of the modish approach provides an upper hand when the model learns thereby enabling it to achieve greater performances than conventional approach and can provide an early diagnosis of pneumothoraces.


####REQUIREMENTS####
Module | Version
------ | -------
pandas | 0.25.3
numpy | 1.18.2
scikit-learn | 0.22.2.post1
opencv | 4.2.0
tensorflow | 2.1.0
keras | 2.3.1


#####TEST IMAGE#####
![Test image](https://github.com/shreyasms17/Pneumothorax-Segmentation/tree/main/src/resources/input/test_img1.png)

#####PREDICTTION#####
![Prediction](https://github.com/shreyasms17/Pneumothorax-Segmentation/tree/main/src/resources/saved_output/result.jpeg)



_This is an implementation of the paper **PNEUMOTHORAX SEGMENTATION** presented at [INOCON 2020](http://inoconf.org/)._
