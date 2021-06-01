# Vehicle-detection-and-classification-in-ATR-Dataset
Using ATR Dataset(Infrared dataset with various sizes of objects at different ranges at which data is being captured and with high level of clutter per target ratio, the problem is quiet challenging.  Used UNet shaped 3DCNN fed with the difference in consecutive image as input. The temporal dimensionality resulted in providing better output.  IoU of region detected by UNet and actual detection marks the performance of the model. The detected region is then fed to the ResNeXt model for classifying to various vehicle classes(10 Classes of Army vehicles'  Used a resneXt for classification after detection in 10 classes of the object

# Requirements
pip install -r requirement.txt

# Use following code to test the model

python main.py --viz = True
