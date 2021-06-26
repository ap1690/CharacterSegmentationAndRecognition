# CharacterSegmentationAndRecognition

Main Approch As Follows
1) Input Image
2) Text/Character Segmentation 
3) Denoising and Upscaling with interpolation
4) Character Recognition
5) Added to CSV

# Solution
The Dataset Contains 50 Mixed Images from Car,Two Wheelers and Posters. 
Some images don't need ROI filterings but some images have a lot background for which it need ROI filtering (Mostly The Number Plates From Two Wheelers)
Here is the reference

![Need ROI Filtering](https://github.com/ap1690/CharacterSegmentationAndRecognition/blob/master/asset/Screenshot_16.png)

For that purpose i applied applied Warped Planar Object Detection Network .WPOD-NET searches for License Plates and
regresses one affine transformation for detection detection, allowing a rectification of the LP area to a rectangle resembling a frontal view
<pre>
scripts/Wpod_net_detection_Contour_segmentation.py
scripts/lpdr.py
weights/wpod_net_update1.h5
</pre>
# Results for ROI segmentation 

![Result](https://github.com/ap1690/CharacterSegmentationAndRecognition/blob/master/asset/Screenshot_17.png)

Have then applied opencv customised thresholding,Connected Components,contour detection and selection . 
<pre>
scripts/Wpod_net_detection_Contour_segmentation.py
scripts/glare.py
scripts/Character-segmentation-Contours.py
scripts/deblur.py
scripts/transform.py
</pre>
![Result](https://github.com/ap1690/CharacterSegmentationAndRecognition/blob/master/asset/masked_samples.png)

Since Dataset is mixed so it became harder for these naive methods to generalize .The extension to this can be applying KNN models to characterise similar characters into a bucket and then assigning the value of the character based on the best image. Since the data was less so scope became less to propogate this way.

# Findings On Dataset
While doing the explore study in terms of understanding the data distribution correctly including Contrast Adjustment Needs,occlusion,Spatial size for building the accurate approches for recognition. Based on 50 images have concluded that ROI filtering and Text/Character Segmentation can be done by using Text Detectiom

# Final Approch
Hence *Applied KerasOCR* for Character Detection and using Wraped Perspective Transform for treating the tilt images.

Here is how results are

![Result](https://github.com/ap1690/CharacterSegmentationAndRecognition/blob/master/asset/Screenshot_8.png)
![Result](https://github.com/ap1690/CharacterSegmentationAndRecognition/blob/master/asset/Screenshot_9.png)
![Result](https://github.com/ap1690/CharacterSegmentationAndRecognition/blob/master/asset/Screenshot_10.png)
![Result](https://github.com/ap1690/CharacterSegmentationAndRecognition/blob/master/asset/Screenshot_11.png)


KerasOCR is a highspeed ocr . Which produced good result interms of Text/Character Detection but results were not sufficient for Recognition so i employed Tesseract For Final Recognition. 

I modified the approch and it as follows
image-->kerasOCR-->Denoising-->Upscaling(Interpolation)-->Debluring-->Tesseract-->Result
<pre>
scripts/Main.ipynb
</pre>

![Result](https://github.com/ap1690/CharacterSegmentationAndRecognition/blob/master/asset/Screenshot_1.png)

![Result](https://github.com/ap1690/CharacterSegmentationAndRecognition/blob/master/asset/Screenshot_2.png)

![Result](https://github.com/ap1690/CharacterSegmentationAndRecognition/blob/master/asset/Screenshot_3.png)

![Result](https://github.com/ap1690/CharacterSegmentationAndRecognition/blob/master/asset/Screenshot_4.png)

![Result](https://github.com/ap1690/CharacterSegmentationAndRecognition/blob/master/asset/Screenshot_5.png)

![Result](https://github.com/ap1690/CharacterSegmentationAndRecognition/blob/master/asset/Screenshot_6.png)

![Result](https://github.com/ap1690/CharacterSegmentationAndRecognition/blob/master/asset/Screenshot_7.png)


Have Wroted The Results Back In CSV
<pre>
result.csv
</pre>

# Further Improvements
I have found out that in some samples our approch is sometimes predicting Last 4 numbers in LC as characters because of distortion and low resolution and for the real world data which can be more noiser. So have found out SVHN housing number dataset which has more robust,occluded,low resolution and noisy data samples .Which will be perfect for recognising number

Samples:


![Result](https://github.com/ap1690/CharacterSegmentationAndRecognition/blob/master/asset/Screenshot_12.png)
![Result](https://github.com/ap1690/CharacterSegmentationAndRecognition/blob/master/asset/Screenshot_13.png)

Have trained my_model.h5 on this data 
<pre>scripts/digit_recognition_svhn.ipynb
weights/my_model.h5
</pre>

Here are the supporting results

![Result](https://github.com/ap1690/CharacterSegmentationAndRecognition/blob/master/asset/Screenshot_15.png)
