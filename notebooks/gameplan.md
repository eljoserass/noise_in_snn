# gameplan

## getting the data

- download a mini slice of images from my bucket in dsec, try to make it a bit parallel right away. for reference on how to access see dsec-det/scripts/dsec_to_r2.py
    the mini slice can come from here {here goes the r2 bucket url}/neuromorph-noise-data/dsec/train/zurich_city_02_b/ 
    it contains left and right calibrations disorted and rectified, for displaying and for the rest of the experiments we can focus on rectified, and right only
    still, load those folders to explore the data, one experiment can be applying the calibration to distorted, displayuig and see if they look the same after the calibration
- display the sample images and event, with the bounding boxes of the object detection labels
- display some insights of the data (eg, number of frames, event, size of the image etc)

## processing the data
- install v2e (https://github.com/SensorsINI/v2e)
- prepare the sample displayed before to the format v2e expects
- generate events of the whole sequence, no perturbations yet
- display generated events

## tinker a little
- display the events frm the dataset with generate events. do they look similar?
- apply noise to the rgb (https://github.com/bethgelab/imagecorruptions) use only one severity and noise for now, eg rain
    display the noised rgb, then generate events on the noised rgb, display the noised simuated events
- apply shot/gaussian noise (whatever noise is most similar to the one v2e generates by using their parameter) on rgb with the same library, now simulate events using CLEAN rgb, but this time use the noise parameter that shold resemple the intensity of noise on the smiulated one
